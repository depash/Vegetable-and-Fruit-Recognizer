'use client'; // Required for using useState and browser event handlers

import { useState } from 'react';

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

  // 1. Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null); // Clear previous prediction
      setError(null);
    }
  };

  // 2. Handle the upload and fetch request
  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Create FormData object to send the file
    const formData = new FormData();
    // The key 'file' must match what the Flask backend is expecting (request.files['file'])
    formData.append('file', selectedFile);

    try {
      // NOTE: Replace the URL with your actual backend URL if it's different
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData, // FormData handles the Content-Type automatically
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
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">🍎 Vegetable and Fruit Classifier 🥦</h1>

      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-lg">
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            1. Select Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none p-2"
          />
        </div>

        <button
          onClick={handleUpload}
          disabled={!selectedFile || isLoading}
          className={`w-full py-3 px-4 rounded-lg text-white font-semibold transition duration-300 
            ${
              selectedFile && !isLoading
                ? 'bg-green-600 hover:bg-green-700'
                : 'bg-green-400 cursor-not-allowed'
            }`}
        >
          {isLoading ? 'Processing Image...' : '2. Get Prediction'}
        </button>

        <hr className="my-8 border-t border-gray-200" />

        <div className="text-center">
          {error && (
            <p className="text-red-600 font-medium bg-red-100 p-3 rounded-md border border-red-300">
              Error: {error}
            </p>
          )}

          {prediction && (
            <div className="bg-blue-100 p-5 rounded-lg border border-blue-300">
              <p className="text-lg text-gray-600 mb-2">Prediction Result:</p>
              <p className="text-3xl font-extrabold text-blue-800 uppercase">
                {prediction}
              </p>
            </div>
          )}

          {!selectedFile && !isLoading && !error && !prediction && (
            <p className="text-gray-500 italic">
              Awaiting file selection and prediction...
            </p>
          )}
        </div>
      </div>

      {selectedFile && (
        <div className="mt-8">
          <h3 className="text-xl font-semibold mb-2 text-gray-700">Image Preview:</h3>
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Selected Preview"
            className="max-w-xs max-h-48 rounded-lg shadow-md object-cover"
          />
        </div>
      )}
    </div>
  );
}