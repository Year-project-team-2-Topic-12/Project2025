import { useState, ChangeEvent, FormEvent } from 'react';
import './App.css';

interface PredictionResponse {
  filename: string;
  prediction?: string | number; confidence?: number;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      alert("Пожалуйста, выберите файл!");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/forward', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Ошибка сервера: ${response.statusText}`);
      }

      const data = (await response.json()) as PredictionResponse;
      setPrediction(data);

    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Неизвестная ошибка');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>MURA ML Upload</h1>

      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-wrapper">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
        </div>

        <button type="submit" disabled={!selectedFile || loading}>
          {loading ? 'Загрузка...' : 'Отправить на анализ'}
        </button>
      </form>

      {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}

      {prediction && (
        <div className="result-block">
          <h2>Результат анализа:</h2>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;