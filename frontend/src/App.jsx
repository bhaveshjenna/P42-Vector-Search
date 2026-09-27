import { useState } from 'react'
import './index.css'

function App() {
  const [activeTab, setActiveTab] = useState('text') // 'text' or 'image'
  const [textQuery, setTextQuery] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleTextSearch = async (e) => {
    e.preventDefault()
    if (!textQuery.trim()) return

    setIsLoading(true)
    setError(null)
    setResults([])

    try {
      const res = await fetch('http://localhost:8000/search/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: textQuery }),
      })
      
      if (!res.ok) throw new Error('Search request failed')
      
      const data = await res.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleImageSearch = async (e) => {
    e.preventDefault()
    if (!imageFile) return

    setIsLoading(true)
    setError(null)
    setResults([])

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const res = await fetch('http://localhost:8000/search/image', {
        method: 'POST',
        body: formData,
      })
      
      if (!res.ok) throw new Error('Search request failed')
      
      const data = await res.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen p-8">
      <header className="max-w-4xl mx-auto mb-12 text-center">
        <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 mb-4">P42</h1>
        <p className="text-xl text-gray-600">Multimodal Vector Search Engine</p>
      </header>

      <main className="max-w-4xl mx-auto">
        {/* Tabs */}
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setActiveTab('text')}
            className={`px-6 py-2 rounded-full font-medium transition-colors ${
              activeTab === 'text'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Text Search
          </button>
          <button
            onClick={() => setActiveTab('image')}
            className={`px-6 py-2 rounded-full font-medium transition-colors ${
              activeTab === 'image'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Image Search
          </button>
        </div>

        {/* Search Forms */}
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 mb-12">
          {activeTab === 'text' ? (
            <form onSubmit={handleTextSearch} className="flex gap-4">
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="Describe what you're looking for..."
                className="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                disabled={isLoading}
                className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50"
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleImageSearch} className="flex flex-col items-center gap-6">
              <div className="w-full max-w-md">
                <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {imagePreview ? (
                      <img src={imagePreview} alt="Preview" className="h-32 object-contain" />
                    ) : (
                      <>
                        <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                        <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                        <p className="text-xs text-gray-500">JPG, PNG, WEBP</p>
                      </>
                    )}
                  </div>
                  <input type="file" className="hidden" accept="image/*" onChange={handleImageChange} />
                </label>
              </div>
              <button
                type="submit"
                disabled={isLoading || !imageFile}
                className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50"
              >
                {isLoading ? 'Searching...' : 'Search Similar Images'}
              </button>
            </form>
          )}
        </div>

        {/* Errors */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 text-red-700 rounded-xl text-center">
            {error}
          </div>
        )}

        {/* Results Grid */}
        {results.length > 0 && (
          <div>
            <h2 className="text-2xl font-bold mb-6 text-gray-800">Top Matches</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {results.map((result) => (
                <div key={result.id} className="bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                  <div className="aspect-square bg-gray-100 relative">
                    <img 
                      src={result.path} 
                      alt={`Result ${result.id}`}
                      className="absolute inset-0 w-full h-full object-cover"
                      loading="lazy"
                    />
                  </div>
                  <div className="p-4">
                    <div className="flex justify-between items-center text-sm text-gray-500">
                      <span>ID: {result.id}</span>
                      <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded-md font-medium text-xs">
                        Score: {result.score.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
