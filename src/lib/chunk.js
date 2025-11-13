import { LocationQueue } from './util'
import ndarray from 'ndarray'

export function Chunk(noa, requestID, ci, cj, ck, size, dataArray, fillVoxelID = -1) {
    this.noa = noa
    this.isDisposed = false
    this.userData = null

    this.requestID = requestID
    this.voxels = dataArray
    this.size = size
    this.i = ci; this.j = cj; this.k = ck
    this.x = ci * size; this.y = cj * size; this.z = ck * size
    this.pos = [this.x, this.y, this.z]

    // mesh flags
    this._terrainDirty = false
    this._objectsDirty = false
    this._pendingRemesh = false
    this._isFull = false
    this._isEmpty = false

    this._wholeLayerVoxel = new Int16Array(size).fill(-1)
    if (fillVoxelID >= 0) {
        this.voxels.data.fill(fillVoxelID)
        this._wholeLayerVoxel.fill(fillVoxelID)
    }

    // neighbor array
    const narr = new Array(27).fill(null)
    this._neighbors = ndarray(narr, [3, 3, 3]).lo(1, 1, 1)
    this._neighbors.set(0, 0, 0, this)

    this._neighborCount = 0
    this._timesMeshed = 0

    this._blockHandlerLocs = new LocationQueue()

    noa._terrainMesher.initChunk(this)
    noa._objectMesher.initChunk(this)

    scanVoxelData(this)
}

Chunk._createVoxelArray = function (size) {
    return ndarray(new Uint16Array(size * size * size), [size, size, size])
}

Chunk.prototype._updateVoxelArray = function (dataArray, fillVoxelID = -1) {
    callAllBlockHandlers(this, 'onUnload')
    const noa = this.noa
    noa._objectMesher.disposeChunk(this)
    noa._terrainMesher.disposeChunk(this)

    this.voxels = dataArray
    this._terrainDirty = this._objectsDirty = false
    this._blockHandlerLocs.empty()
    noa._objectMesher.initChunk(this)
    noa._terrainMesher.initChunk(this)

    const fillVal = (fillVoxelID >= 0) ? fillVoxelID : -1
    this._wholeLayerVoxel.fill(fillVal)

    scanVoxelData(this)
}

Chunk.prototype.get = function (i, j, k) {
    return this.voxels.data[this.voxels.index(i, j, k)]
}

Chunk.prototype.getSolidityAt = function (i, j, k) {
    return this.noa.registry._solidityLookup[
        this.voxels.data[this.voxels.index(i, j, k)]
    ]
}

Chunk.prototype.set = function (i, j, k, newID) {
    const data = this.voxels.data
    const idx = this.voxels.index(i, j, k)
    const oldID = data[idx]
    if (newID === oldID) return

    const noa = this.noa
    const reg = noa.registry
    const solid = reg._solidityLookup
    const opaque = reg._opacityLookup
    const object = reg._objectLookup
    const handler = reg._blockHandlerLookup

    data[idx] = newID

    if (!opaque[newID]) this._isFull = false
    if (newID !== 0) this._isEmpty = false
    if (this._wholeLayerVoxel[j] !== newID) this._wholeLayerVoxel[j] = -1

    const hold = handler[oldID]
    const hnew = handler[newID]
    if (hold) callBlockHandler(this, hold, 'onUnset', i, j, k)
    if (hnew) {
        callBlockHandler(this, hnew, 'onSet', i, j, k)
        this._blockHandlerLocs.add(i, j, k)
    } else {
        this._blockHandlerLocs.remove(i, j, k)
    }

    const objMesher = noa._objectMesher
    if (object[oldID]) objMesher.setObjectBlock(this, 0, i, j, k)
    if (object[newID]) objMesher.setObjectBlock(this, newID, i, j, k)

    const solidityChanged = solid[oldID] !== solid[newID]
    const opacityChanged = opaque[oldID] !== opaque[newID]
    const wasTerrain = !object[oldID] && oldID !== 0
    const nowTerrain = !object[newID] && newID !== 0

    if (object[oldID] || object[newID]) this._objectsDirty = true
    if (solidityChanged || opacityChanged || wasTerrain || nowTerrain)
        this._terrainDirty = true

    if ((this._terrainDirty || this._objectsDirty) && !this._pendingRemesh) {
        this._pendingRemesh = true
        noa.world._queueChunkForRemesh(this)
    }

    // Neighbor updates only when at edges
    if (solidityChanged || opacityChanged) {
        const edge = this.size - 1
        if (i === 0 || j === 0 || k === 0 || i === edge || j === edge || k === edge) {
            for (let ni = -1; ni <= 1; ni++) {
                for (let nj = -1; nj <= 1; nj++) {
                    for (let nk = -1; nk <= 1; nk++) {
                        if ((ni | nj | nk) === 0) continue
                        const nab = this._neighbors.get(ni, nj, nk)
                        if (nab && !nab._pendingRemesh) {
                            nab._terrainDirty = true
                            nab._pendingRemesh = true
                            noa.world._queueChunkForRemesh(nab)
                        }
                    }
                }
            }
        }
    }
}

function callBlockHandler(chunk, handlers, type, i, j, k) {
    if (!handlers) return
    const fn = handlers[type]
    if (fn) fn(chunk.x + i, chunk.y + j, chunk.z + k)
}

Chunk.prototype.updateMeshes = function () {
    const noa = this.noa
    if (this._terrainDirty) {
        noa._terrainMesher.meshChunk(this)
        this._timesMeshed++
        this._terrainDirty = false
    }
    if (this._objectsDirty) {
        noa._objectMesher.buildObjectMeshes()
        this._objectsDirty = false
    }
    this._pendingRemesh = false
}

/* Efficient voxel scan */
function scanVoxelData(chunk) {
    const voxels = chunk.voxels
    const data = voxels.data
    const size = voxels.shape[0]
    const reg = chunk.noa.registry
    const opaque = reg._opacityLookup
    const plain = reg._blockIsPlainLookup
    const object = reg._objectLookup
    const handler = reg._blockHandlerLookup
    const objMesher = chunk.noa._objectMesher

    let fullyOpaque = true, fullyAir = true

    for (let j = 0; j < size; j++) {
        let constantID = data[voxels.index(0, j, 0)]
        let layerConst = true

        for (let i = 0; i < size; i++) {
            let baseIdx = voxels.index(i, j, 0)
            for (let k = 0; k < size; k++, baseIdx++) {
                const id = data[baseIdx]
                if (id === 0) { fullyOpaque = false; continue }
                if (plain[id]) { fullyAir = false; continue }

                fullyOpaque &&= opaque[id]
                fullyAir = false

                if (object[id]) {
                    objMesher.setObjectBlock(chunk, id, i, j, k)
                    chunk._objectsDirty = true
                }

                const h = handler[id]
                if (h) {
                    chunk._blockHandlerLocs.add(i, j, k)
                    callBlockHandler(chunk, h, 'onLoad', i, j, k)
                }

                if (layerConst && id !== constantID) layerConst = false
            }
        }

        chunk._wholeLayerVoxel[j] = layerConst ? constantID : -1
    }

    chunk._isFull = fullyOpaque
    chunk._isEmpty = fullyAir
    chunk._terrainDirty = !chunk._isEmpty
}

Chunk.prototype.dispose = function () {
    callAllBlockHandlers(this, 'onUnload')
    this._blockHandlerLocs.empty()

    const noa = this.noa
    noa._objectMesher.disposeChunk(this)
    noa._terrainMesher.disposeChunk(this)

    this.voxels.data = null
    this.voxels = null
    this._neighbors.data = null
    this._neighbors = null

    this.isDisposed = true
}

function callAllBlockHandlers(chunk, type) {
    const voxels = chunk.voxels
    const handlerLookup = chunk.noa.registry._blockHandlerLookup
    const arr = chunk._blockHandlerLocs.arr
    const len = arr.length
    const data = voxels.data

    for (let n = 0; n < len; n++) {
        const [i, j, k] = arr[n]
        const id = data[voxels.index(i, j, k)]
        callBlockHandler(chunk, handlerLookup[id], type, i, j, k)
    }
}
