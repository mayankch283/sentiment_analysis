import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
  };

  const handleFileSubmit = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadResponse = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const analysisResponse = await axios.post('http://localhost:5000/analyze', {
        filename: uploadResponse.data.filename,
      });

      setResult(analysisResponse.data);
    } catch (error) {
      console.error('Error uploading or analyzing file:', error);
    }
  };

  const handleUrlSubmit = async () => {
    try {
      const analysisResponse = await axios.post('http://localhost:5000/analyze-url', {
        url,
      });

      setResult(analysisResponse.data);
    } catch (error) {
      console.error('Error analyzing URL:', error);
    }
  };

  return (
    <div>
      <h1>Sentiment Analysis Platform</h1>
      <div>
        <h2>Upload File</h2>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleFileSubmit}>Upload and Analyze</button>
      </div>
      <div>
        <h2>Submit URL</h2>
        <input
          type="text"
          value={url}
          onChange={handleUrlChange}
          placeholder="Enter URL"
        />
        <button onClick={handleUrlSubmit}>Analyze URL</button>
      </div>
      {result && (
        <div>
          <h2>Analysis Result</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
