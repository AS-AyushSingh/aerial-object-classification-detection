import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Predict from './pages/Predict'
import Header from './components/Header'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </main>
      <footer className="footer">
        <span>© {new Date().getFullYear()} Aerial Object Classification</span>
      </footer>
    </div>
  )
}
