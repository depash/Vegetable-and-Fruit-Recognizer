'use client'; 

import { useState } from 'react';
import styles from './Home.module.css'; // <-- Import the module CSS

// Define the type for the prediction result
interface PredictionResponse {
  status: string;
  prediction: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    // IMPORTANT: In production, change the URL from 'http://127.0.0.1:5000/predict' to just '/api/predict'
    // after deploying your Flask backend to the Vercel Serverless Function (as per the earlier guidance).
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      
      if (data.status === 'success') {
        setPrediction(data.prediction);
      } else {
        setError(`Prediction failed: ${data.prediction}`);
      }
      
    } catch (err) {
      console.error('Upload Error:', err);
      setError('Failed to connect to the prediction server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // Replaced all Tailwind classes with semantic CSS Module classes
    <div className={styles.container}>
      <h1 className={styles.title}>🍎 Vegetable and Fruit Classifier 🥦</h1>

      <div className={styles.card}>
        
        <div className={styles.inputGroup}>
          <label htmlFor="file-upload" className={styles.label}>
            1. Select Image
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className={styles.fileInput}
          />
        </div>

        <button
          onClick={handleUpload}
          disabled={!selectedFile || isLoading}
          className={`${styles.button} ${(!selectedFile || isLoading) ? styles.buttonDisabled : ''}`}
        >
          {isLoading ? 'Processing Image...' : '2. Get Prediction'}
        </button>

        <hr className={styles.divider} />

        <div className={styles.resultArea}>
          {error && (
            <p className={`${styles.message} ${styles.messageError}`}>
              Error: {error}
            </p>
          )}

          {prediction && (
            <div className={`${styles.message} ${styles.messageSuccess}`}>
              <p className={styles.resultLabel}>Prediction Result:</p>
              <p className={styles.resultText}>
                {prediction}
              </p>
            </div>
          )}

          {!selectedFile && !isLoading && !error && !prediction && (
            <p className={styles.messagePlaceholder}>
              Awaiting file selection and prediction...
            </p>
          )}
        </div>
      </div>

      {selectedFile && (
        <div className={styles.imagePreviewContainer}>
          <h3 className={styles.previewTitle}>Image Preview:</h3>
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Selected Preview"
            className={styles.imagePreview}
          />
        </div>
      )}
    </div>
  );
}