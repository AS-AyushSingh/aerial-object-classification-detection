import { Link, NavLink } from 'react-router-dom'
import './Header.css'

export default function Header() {
  return (
    <header className="header">
      <div className="header__brand">
        <Link to="/" className="header__logo">
          Aerial<span className="header__logo--accent">AI</span>
        </Link>
        <p className="header__subtitle">Bird vs Drone Classification</p>
      </div>
      <nav className="header__nav" aria-label="Main navigation">
        <NavLink to="/" end className={({ isActive }) => (isActive ? 'nav__link nav__link--active' : 'nav__link')}>
          Home
        </NavLink>
        <NavLink to="/predict" className={({ isActive }) => (isActive ? 'nav__link nav__link--active' : 'nav__link')}>
          Predict
        </NavLink>
      </nav>
    </header>
  )
}
