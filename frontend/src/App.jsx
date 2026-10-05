import { useState } from 'react'
import './index.css'

const API_URL = import.meta.env.VITE_API_URL || ''

function App() {
  const [mode, setMode] = useState('text-to-image')
  const [textQuery, setTextQuery] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [results, setResults] = useState([])
  const [latency, setLatency] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedItem, setSelectedItem] = useState(null)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => setImagePreview(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResults([])
    setLatency(null)
    try {
      let res
      if (mode === 'text-to-image') {
        res = await fetch(`${API_URL}/search/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: textQuery }),
        })
      } else if (mode === 'image-to-image') {
        const fd = new FormData()
        fd.append('file', imageFile)
        res = await fetch(`${API_URL}/search/image`, { method: 'POST', body: fd })
      } else {
        const fd = new FormData()
        fd.append('file', imageFile)
        res = await fetch(`${API_URL}/search/image-to-text`, { method: 'POST', body: fd })
      }
      if (!res.ok) throw new Error(res.statusText)
      const data = await res.json()
      setResults(data.results)
      setLatency(data.total_latency_ms)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const modes = [
    { id: 'text-to-image', label: 'Text → Image' },
    { id: 'image-to-image', label: 'Image → Image' },
    { id: 'image-to-text', label: 'Image → Text' },
  ]

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">

      {/* NAV */}
      <nav className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <span className="font-bold text-white text-lg">Multimodal Image Search</span>
          <span className="ml-3 text-xs text-gray-500 font-mono">CLIP + FAISS · Flickr30k</span>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-10">

        {/* MODE TABS */}
        <div className="flex gap-1 mb-8 border-b border-gray-800">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => { setMode(m.id); setResults([]) }}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                mode === m.id
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* SEARCH */}
        <form onSubmit={handleSearch} className="mb-10">
          {mode === 'text-to-image' ? (
            <div className="flex gap-3">
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="e.g. a dog running on the beach"
                className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
              <button
                type="submit"
                disabled={isLoading || !textQuery}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-medium px-6 py-3 rounded-lg transition-colors"
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          ) : (
            <div className="flex gap-3 items-center">
              <label className="flex-1 flex items-center gap-3 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 cursor-pointer hover:border-gray-600 transition-colors">
                {imagePreview
                  ? <img src={imagePreview} className="h-8 w-8 rounded object-cover" />
                  : <span className="text-gray-600 text-sm">📁</span>
                }
                <span className="text-sm text-gray-400">{imageFile ? imageFile.name : 'Click to upload an image'}</span>
                <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
              </label>
              <button
                type="submit"
                disabled={isLoading || !imageFile}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-medium px-6 py-3 rounded-lg transition-colors"
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          )}
          {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
        </form>

        {/* RESULTS */}
        {latency !== null && (
          <p className="text-xs text-gray-600 font-mono mb-4">
            {results.length} results · {latency.toFixed(1)}ms total
          </p>
        )}

        <div className="flex flex-col gap-3">
          {results.map((r, idx) => (
            <div
              key={idx}
              className="flex gap-4 bg-gray-900 border border-gray-800 rounded-xl p-3 hover:border-gray-700 transition-colors"
            >
              {/* Thumbnail — click to fullscreen */}
              <img
                src={`${API_URL}${r.image_url}`}
                alt={r.filename}
                onClick={() => setSelectedItem(r)}
                className="w-24 h-24 object-cover rounded-lg flex-shrink-0 cursor-pointer"
              />

              {/* Info */}
              <div className="flex flex-col justify-center gap-2 flex-1 min-w-0">
                {mode === 'image-to-text' && r.matched_text && (
                  <p className="text-xs text-indigo-400 italic truncate">"{r.matched_text}"</p>
                )}
                <p className="text-sm text-gray-300 truncate font-medium">{r.filename}</p>
                <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs font-mono text-gray-500">
                  <span>Cosine similarity: <span className="text-green-400">{r.similarity_pct}</span></span>
                  <span>FAISS ID: <span className="text-indigo-400">{r.faiss_id}</span></span>
                  <span>Latency: <span className="text-amber-400">{r.latency_ms.toFixed(1)}ms</span></span>
                  <span className="truncate max-w-xs" title={`[${r.embedding_preview}]`}>
                    Emb: <span className="text-gray-400">[{r.embedding_preview}]</span>
                  </span>
                </div>
              </div>

              {/* Rank badge */}
              <div className="flex-shrink-0 flex items-center justify-center w-8 text-xs font-mono text-gray-700">
                #{idx + 1}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* FULLSCREEN MODAL */}
      {selectedItem && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex"
          onClick={() => setSelectedItem(null)}
        >
          {/* Image */}
          <div className="flex-1 flex items-center justify-center p-8" onClick={(e) => e.stopPropagation()}>
            <img
              src={`${API_URL}${selectedItem.image_url}`}
              alt={selectedItem.filename}
              className="max-w-full max-h-screen object-contain rounded-lg"
            />
          </div>

          {/* Sidebar */}
          <div
            className="w-80 bg-gray-950 border-l border-gray-800 flex flex-col p-6 overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSelectedItem(null)}
              className="self-end text-gray-600 hover:text-gray-300 mb-6 text-sm"
            >
              ✕ Close
            </button>

            <p className="text-sm font-medium text-gray-300 mb-4 break-all">{selectedItem.filename}</p>

            {/* Stats */}
            <div className="flex flex-col gap-3 text-xs font-mono mb-6 border border-gray-800 rounded-lg p-4 bg-gray-900">
              <div className="flex justify-between">
                <span className="text-gray-500">Cosine Similarity</span>
                <span className="text-green-400 font-bold">{selectedItem.similarity_pct}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">FAISS Index ID</span>
                <span className="text-indigo-400">{selectedItem.faiss_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Inference Latency</span>
                <span className="text-amber-400">{selectedItem.latency_ms.toFixed(1)}ms</span>
              </div>
              <div className="pt-2 border-t border-gray-800 text-gray-600 break-all">
                [{selectedItem.embedding_preview}]
              </div>
            </div>

            {/* Captions */}
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Ground Truth Captions</p>
            <ul className="flex flex-col gap-2">
              {selectedItem.ground_truth_captions?.length > 0
                ? selectedItem.ground_truth_captions.map((cap, i) => (
                    <li key={i} className="text-xs text-gray-400 leading-relaxed bg-gray-900 border border-gray-800 rounded-lg p-3">
                      {cap}
                    </li>
                  ))
                : <li className="text-xs text-gray-600 italic">No captions available.</li>
              }
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
