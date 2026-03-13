import { useEffect, useMemo, useRef, useState } from 'react'
import './Predict.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api'
const API_ENDPOINT = `${API_BASE}/predict`

function formatPercent(v) {
  return `${Math.round(v * 100)}%`
}

export default function Predict() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!file) return

    // reset previous result when selecting a new file
    setResult(null)
  }, [file])

  const canSubmit = useMemo(() => !!file && !loading, [file, loading])

  const predictionLabel = useMemo(() => result?.prediction ?? '—', [result])
  const confidence = useMemo(() => (result?.confidence ?? 0), [result])

  function handleFileChange(event) {
    setError(null)
    const selected = event.target.files?.[0]
    if (!selected) return
    setFile(selected)
  }

  function handleDragOver(event) {
    event.preventDefault()
    setDragOver(true)
  }

  function handleDragLeave(event) {
    event.preventDefault()
    setDragOver(false)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragOver(false)
    const files = event.dataTransfer.files
    if (files.length > 0) {
      setError(null)
      setFile(files[0])
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)
    setResult(null)

    if (!file) {
      setError('Please select an image before submitting.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    try {
      setLoading(true)
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.error || 'Unexpected server response')
      }
      setResult(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <section className="predict">
      <header className="predict__header">
        <h1 className="predict__title">Drone / Bird Detection</h1>
        <p className="predict__subtitle">
          Upload an aerial image and see the model's confidence scores.
        </p>
      </header>

      <section
        className={`predict__upload ${dragOver ? 'predict__upload--drag' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          ref={inputRef}
          style={{ display: 'none' }}
        />
        <label htmlFor="file-input" className="predict__uploadLabel">
          {preview ? (
            <img src={preview} alt="Uploaded image" className="predict__preview" />
          ) : (
            <div className="predict__uploadPrompt">
              <div className="predict__uploadIcon">📁</div>
              <div className="predict__uploadText">
                <strong>Drop an image here</strong> or click to select
              </div>
            </div>
          )}
        </label>

        <div className="predict__actions">
          <button className="button" type="button" onClick={handleSubmit} disabled={!canSubmit}>
            {loading ? 'Predicting…' : 'Predict'}
          </button>
          <button
            className="button button--ghost"
            type="button"
            onClick={handleReset}
            disabled={loading && !file}
          >
            Reset
          </button>
        </div>

        {error ? <p className="predict__error">{error}</p> : null}
      </section>

      <section
        className="predict__scores"
        style={{ display: result ? 'block' : 'none' }}
      >
        <h2 className="predict__sectionTitle">Results</h2>

        <div className="predict__resultCard">
          <div className="predict__resultItem">
            <span className="predict__resultLabel">Prediction:</span>
            <span className="predict__resultValue">{predictionLabel}</span>
          </div>
          <div className="predict__resultItem">
            <span className="predict__resultLabel">Confidence:</span>
            <span className="predict__resultValue">{formatPercent(confidence)}</span>
          </div>
        </div>

        <div className="predict__probabilities">
          <h3 className="predict__probabilitiesTitle">Class Probabilities</h3>
          {result?.probabilities &&
            Object.entries(result.probabilities).map(([label, value]) => (
              <div key={label} className="predict__probability">
                <div className="predict__probabilityHeader">
                  <span className="predict__probabilityLabel">{label}</span>
                  <span className="predict__probabilityValue">{formatPercent(value)}</span>
                </div>
                <div className="predict__probabilityBar">
                  <div
                    className={`predict__probabilityFill predict__probabilityFill--${label}`}
                    style={{ width: `${Math.max(6, value * 100)}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      </section>
    </section>
  )
}
