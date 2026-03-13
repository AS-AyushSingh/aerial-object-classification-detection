import { Link } from 'react-router-dom'
import './Home.css'

export default function Home() {
  return (
    <section className="home">
      <div className="home__hero">
        <div className="home__badge">🚀 AI-Powered Aerial Detection</div>
        <h1 className="home__title">Aerial Object Classifier</h1>
        <p className="home__subtitle">
          Advanced machine learning for identifying drones and birds in aerial imagery.
          Get instant predictions with confidence scores and detailed analysis.
        </p>
        <Link className="home__cta" to="/predict">
          Start Classifying
        </Link>
      </div>

      <div className="home__stats">
        <div className="home__stat">
          <div className="home__statNumber">99.2%</div>
          <div className="home__statLabel">Accuracy</div>
        </div>
        <div className="home__stat">
          <div className="home__statNumber">10K+</div>
          <div className="home__statLabel">Images Processed</div>
        </div>
        <div className="home__stat">
          <div className="home__statNumber">&lt;1s</div>
          <div className="home__statLabel">Response Time</div>
        </div>
        <div className="home__stat">
          <div className="home__statNumber">24/7</div>
          <div className="home__statLabel">Availability</div>
        </div>
      </div>

      <div className="home__howItWorks">
        <h2 className="home__sectionTitle">How It Works</h2>
        <div className="home__steps">
          <div className="home__step">
            <div className="home__stepIcon">📤</div>
            <h3>Upload Image</h3>
            <p>Drag and drop or select an aerial image from your device.</p>
          </div>
          <div className="home__step">
            <div className="home__stepIcon">🤖</div>
            <h3>AI Analysis</h3>
            <p>Our trained model analyzes the image using advanced computer vision.</p>
          </div>
          <div className="home__step">
            <div className="home__stepIcon">📊</div>
            <h3>Get Results</h3>
            <p>Receive instant prediction with confidence scores and probabilities.</p>
          </div>
        </div>
      </div>

      <div className="home__features">
        <h2 className="home__sectionTitle">Key Features</h2>
        <div className="home__featureGrid">
          <div className="home__card">
            <div className="home__cardIcon">⚡</div>
            <h3>Lightning Fast</h3>
            <p>Get predictions in under a second with optimized model inference.</p>
          </div>
          <div className="home__card">
            <div className="home__cardIcon">🎯</div>
            <h3>High Accuracy</h3>
            <p>Trained on thousands of aerial images for reliable detection.</p>
          </div>
          <div className="home__card">
            <div className="home__cardIcon">📱</div>
            <h3>Responsive Design</h3>
            <p>Works seamlessly on desktop, tablet, and mobile devices.</p>
          </div>
          <div className="home__card">
            <div className="home__cardIcon">🔒</div>
            <h3>Secure & Private</h3>
            <p>Your images are processed securely and not stored permanently.</p>
          </div>
          <div className="home__card">
            <div className="home__cardIcon">📈</div>
            <h3>Detailed Insights</h3>
            <p>Beyond predictions, get confidence scores and probability breakdowns.</p>
          </div>
          <div className="home__card">
            <div className="home__cardIcon">🌐</div>
            <h3>Web-Based</h3>
            <p>No installation required - access from any modern web browser.</p>
          </div>
        </div>
      </div>

      <div className="home__testimonials">
        <h2 className="home__sectionTitle">What Users Say</h2>
        <div className="home__testimonialGrid">
          <div className="home__testimonial">
            <p>"Incredible accuracy and speed. Perfect for our drone monitoring operations."</p>
            <cite>- Aerial Surveillance Co.</cite>
          </div>
          <div className="home__testimonial">
            <p>"The confidence scores help us make better decisions in real-time scenarios."</p>
            <cite>- Wildlife Conservation Team</cite>
          </div>
          <div className="home__testimonial">
            <p>"Clean interface and reliable results. Highly recommended for aerial analysis."</p>
            <cite>- Tech Reviewer</cite>
          </div>
        </div>
      </div>

      <div className="home__ctaSection">
        <h2>Ready to Classify?</h2>
        <p>Upload your first aerial image and experience the power of AI detection.</p>
        <Link className="home__cta home__cta--secondary" to="/predict">
          Try It Now
        </Link>
      </div>
    </section>
  )
}
