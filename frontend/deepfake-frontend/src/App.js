import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';
import attachedImage from './assets/deepfake-detection-visual.jpg';  // Make sure the image is saved here

const App = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 100 * 1024 * 1024) {
        setError('File size too large. Please select a video under 100MB.');
        return;
      }
      setFile(selectedFile);
      setError('');
      setUploadResult(null);
      setAnalysisResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    setUploading(true);
    setError('');

    try {
      const uploadResponse = await axios.post('http://localhost:5000/api/upload', formData);
      setUploadResult(uploadResponse.data);
      setUploading(false);

      setAnalyzing(true);
      const analyzeResponse = await axios.post(
        `http://localhost:5000/api/analyze/${uploadResponse.data.filename}`
      );
      setAnalysisResult(analyzeResponse.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Please try again.');
    } finally {
      setUploading(false);
      setAnalyzing(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setUploadResult(null);
    setAnalysisResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <span className="logo-text">DeepFake AI</span>
          </div>
          <div className="nav-links">
            <a href="#" className="nav-link active">Home</a>
            <a href="#" className="nav-link">Features</a>
            <a href="#" className="nav-link">Documentation</a>
            <a href="#" className="nav-link">About</a>            
          </div>
          <button className="nav-cta">Start Analysis</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-container">
          <div className="hero-content">
            <h1 className="hero-title">
              The Smarter Way<br />
              to Detect Deepfakes
            </h1>
            <p className="hero-subtitle">
              AI-powered deepfake detection using advanced ResNet-Swish-BiLSTM neural networks that learn and adapt to emerging threats.
            </p>
            <div className="hero-buttons">
              <button className="btn-primary" onClick={() => fileInputRef.current && fileInputRef.current.click()}>
                Start AI Analysis
              </button>
              <button className="btn-secondary">Get Demo</button>
            </div>
          </div>
          <div className="hero-image">
            <img src={attachedImage} alt="Deepfake Detection Visual" />
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="dashboard">
        <div className="dashboard-container">
          <div className="main-content">
            <div className="content-header">
              
              <div className="header-actions">
                <button className="action-btn">Choose Model</button>
              </div>
            </div>

            <div className="task-manager">
              <h2 className="section-title">Deepfake Detector</h2>

              {/* Stats Cards */}
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-number">{analysisResult ? '1' : '0'}</div>
                  <div className="stat-label">+ Videos Analyzed</div>
                  <div className="stat-badge">Active</div>
                </div>

                <div className="stat-card">
                  <div className="stat-number">{analysisResult ? Math.round(analysisResult.confidence * 100) : '0'}</div>
                  <div className="stat-label">+ Confidence Score</div>
                  <div className="stat-badge">In Progress</div>
                </div>

                <div className="stat-card">
                  <div className="stat-number">{analysisResult ? analysisResult.faces_detected : '0'}</div>
                  <div className="stat-label">+ Faces Detected</div>
                  <div className="stat-badge">Complete</div>
                </div>
              </div>

              {/* Upload Section */}
              <div className="upload-section">
                <div className="upload-card">
                  <div className="upload-icon" style={{ animation: 'none' }}>📁</div>
                  <h3>Upload Video for AI Analysis</h3>
                  <p>Advanced neural networks will analyze your video for deepfake patterns</p>
                  <p className="upload-formats">Supports: MP4, AVI, MOV, MKV · Max size: 100MB</p>

                  <input
                    id="video-upload-input"
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />

                  <button
                    className="upload-btn primary"
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  >
                    Choose File
                  </button>

               {file && (
  <div className="file-upload-action-row">
    <div className="file-card-mini">
      <span className="file-card-icon">🎬</span>
      <div>
        <span className="file-card-filename" title={file.name}>{file.name}</span>
        <div className="file-card-meta">
          {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type}
        </div>
      </div>
      <button className="file-card-remove" onClick={resetForm} title="Remove">×</button>
    </div>
    <button
      className="analyze-btn-mini"
      onClick={handleUpload}
      disabled={uploading || analyzing}
    >
      {uploading ? 'Uploading...' : analyzing ? 'AI Analyzing...' : 'Start Analysis'}
    </button>
  </div>
)}





                </div>
              </div>

              {/* Loading State */}
              {(uploading || analyzing) && (
                <div className="loading-section">
                  <div className="loading-spinner"></div>
                  <p>{uploading ? 'Uploading video...' : 'Analyzing for deepfakes...'}</p>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="error-card">
                  <div className="error-icon">⚠️</div>
                  <div className="error-content">
                    <h4>Analysis Error</h4>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {/* Results Section */}
              {analysisResult && (
                <div className="results-section">
                  <div className="results-grid">
                    <div className="result-card main-result">
                      <div className="result-header">
                        <h3>Detection Result</h3>
                       <div className={`status-badge ${analysisResult.prediction}`} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
  {analysisResult.prediction === 'deepfake' ? (
    <>
      <span role="img" aria-label="Warning" style={{ fontSize: '1.2em' }}>⚠️</span>
      <span>DEEPFAKE</span>
    </>
  ) : (
    <>
      <span role="img" aria-label="Check" style={{ fontSize: '1.2em' }}>✅</span>
      <span>Authentic</span>
    </>
  )}
</div>

                      </div>
                      <div className="confidence-score">
                        <div className="score-number">{Math.round(analysisResult.confidence * 100)}%</div>
                        <div className="score-label">Confidence</div>
                      </div>
                      <div className="result-message">{analysisResult.message}</div>
                    </div>

                    <div className="result-card">
                      <h4>Video Analysis</h4>
                      <div className="analysis-details">
  <div className="detail-item">
    <span className="detail-label">Duration:</span>
    <span className="detail-value">{analysisResult.duration}s</span>
  </div>
  <div className="detail-item">
    <span className="detail-label">Frames:</span>
    <span className="detail-value">{analysisResult.total_frames}</span>
  </div>
  <div className="detail-item">
    <span className="detail-label">Faces:</span>
    <span className="detail-value">{analysisResult.faces_detected}</span>
  </div>
  <div className="detail-item">
    <span className="detail-label">Method:</span>
    <span className="detail-value">ResNet-BiLSTM</span>
  </div>
</div>

                    </div>

                    <div className="result-card">
                      <h4>Risk Assessment</h4>
                      <div className={`risk-level ${analysisResult.risk_level}`}>
                        {analysisResult.risk_level.toUpperCase()} RISK
                      </div>
                      <p className="risk-description">
                        {analysisResult.risk_level === 'high'
                          ? 'High probability of manipulation detected'
                          : analysisResult.risk_level === 'medium'
                            ? 'Some suspicious patterns found'
                            : 'Video appears to be authentic'
                        }
                      </p>
                      <button className="btn-secondary" onClick={resetForm}>
                        Analyze Another Video
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer-container">
        <div className="footer-inner">
          <div className="footer-left">
            <h2 className="footer-logo">DeepFake AI</h2>
            <p className="footer-description">Detecting deepfakes with cutting-edge AI technology.</p>
            <p className="footer-copy">© {new Date().getFullYear()} DeepFake AI. All rights reserved.</p>
          </div>
          <div className="footer-right">
            <div className="footer-links-group">
              <h4 className="footer-links-title">Product</h4>
              <a href="#" className="footer-link">Features</a>
              <a href="#" className="footer-link">Pricing</a>
              <a href="#" className="footer-link">Updates</a>
            </div>
            <div className="footer-links-group">
              <h4 className="footer-links-title">Company</h4>
              <a href="#" className="footer-link">About</a>
              <a href="#" className="footer-link">Careers</a>
              <a href="#" className="footer-link">Contact</a>
            </div>
            <div className="footer-links-group">
              <h4 className="footer-links-title">Resources</h4>
              <a href="#" className="footer-link">Blog</a>
              <a href="#" className="footer-link">Help Center</a>
              <a href="#" className="footer-link">Privacy & Terms</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
