import { useState, FormEvent } from 'react';
import './App.css';
import { forwardForwardPost, forwardMultipleForwardMultiplePost } from './client';
import type {
  BodyForwardForwardPost,
  BodyForwardMultipleForwardMultiplePost,
  ForwardForwardPostData,
  ForwardImageResponse,
  ForwardMultipleForwardMultiplePostData,
  PredictionResponse,
} from './client';

type StudyInput = {
  id: string;
  name: string;
  files: File[];
};

function App() {
  const [mode, setMode] = useState<'single' | 'multiple'>('single');
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [studies, setStudies] = useState<StudyInput[]>([
    { id: crypto.randomUUID(), name: 'Study 1', files: [] },
  ]);
  const [debug, setDebug] = useState<boolean>(false);

  const [predictionSingle, setPredictionSingle] = useState<ForwardImageResponse | null>(null);
  const [predictionMultiple, setPredictionMultiple] = useState<PredictionResponse[] | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const updateStudyFiles = (index: number, files: File[]) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, files } : study)));
    setPredictionSingle(null);
    setPredictionMultiple(null);
    setError(null);
  };

  const updateStudyName = (index: number, name: string) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, name } : study)));
  };

  const addStudy = () => {
    setStudies((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: `Study ${prev.length + 1}`, files: [] },
    ]);
  };

  const removeStudy = (index: number) => {
    setStudies((prev) => prev.filter((_, i) => i !== index));
    setPredictionSingle(null);
    setPredictionMultiple(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (mode === 'single') {
      if (!singleFile) {
        alert("Пожалуйста, выберите файл!");
        return;
      }
    } else {
      if (studies.length === 0 || studies.some((study) => study.files.length === 0)) {
        alert("Пожалуйста, добавьте хотя бы один файл в каждую группу!");
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      if (mode === 'single') {
        const body: BodyForwardForwardPost = { image: singleFile as File };
        const headers: ForwardForwardPostData['headers'] = {
          'X-Debug': debug,
        };
        const data = await forwardForwardPost({
          body,
          headers,
          responseStyle: 'data',
          throwOnError: true,
        });
        setPredictionSingle(data);
        setPredictionMultiple(null);
      } else {
        const files: File[] = [];
        const studyIds: string[] = [];
        studies.forEach((study) => {
          study.files.forEach((file) => {
            files.push(file);
            studyIds.push(study.id);
          });
        });

        const body: BodyForwardMultipleForwardMultiplePost = { images: files };
        const headers: ForwardMultipleForwardMultiplePostData['headers'] = {
          'X-Study-Ids': studyIds.join(','),
          'X-Debug': debug,
        };
        const data = await forwardMultipleForwardMultiplePost({
          body,
          headers,
          responseStyle: 'data',
          throwOnError: true,
        });
        setPredictionMultiple(data);
        setPredictionSingle(null);
      }

    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        console.log(err);
        setError(`ошибка: ${err.message || err.detail || String(err)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>MURA ML Upload</h1>

      <form onSubmit={handleSubmit} className="upload-form">
        <div className="mode-toggle">
          <label>
            <input
              type="radio"
              name="mode"
              value="single"
              checked={mode === 'single'}
              onChange={() => {
                setMode('single');
                setPredictionSingle(null);
                setPredictionMultiple(null);
                setError(null);
              }}
            />
            Одна картинка
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="multiple"
              checked={mode === 'multiple'}
              onChange={() => {
                setMode('multiple');
                setPredictionSingle(null);
                setPredictionMultiple(null);
                setError(null);
              }}
            />
            Исследование / несколько исследований
          </label>
        </div>

        {mode === 'single' ? (
          <div className="single-card">
            <label className="file-input">
              <input
                type="file"
                accept="image/*"
                onChange={(event) => {
                  const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
                  setSingleFile(file);
                  setPredictionSingle(null);
                  setPredictionMultiple(null);
                  setError(null);
                }}
              />
              <span>Выбрать файл</span>
            </label>
            <div className="file-hint">
              {singleFile ? singleFile.name : 'Пока нет файла'}
            </div>
          </div>
        ) : (
          <div className="study-grid">
            {studies.map((study, index) => (
              <div className="study-card" key={study.id}>
                <div className="study-header">
                  <input
                    className="study-name"
                    value={study.name}
                    onChange={(event) => updateStudyName(index, event.target.value)}
                    placeholder={`Study ${index + 1}`}
                  />
                  {studies.length > 1 && (
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => removeStudy(index)}
                    >
                      ❌
                    </button>
                  )}
                </div>
                <label className="file-input">
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(event) =>
                      updateStudyFiles(index, event.target.files ? Array.from(event.target.files) : [])
                    }
                  />
                  <span>Выбрать файлы</span>
                </label>
                <div className="file-hint">
                  {study.files.length > 0
                    ? `Файлов: ${study.files.length}`
                    : 'Пока нет файлов'}
                </div>
              </div>
            ))}
            <button type="button" className="add-card" onClick={addStudy}>
              + Добавить study
            </button>
          </div>
        )}

        <div className="form-actions">
          <button
            type="submit"
            disabled={
              loading ||
              (mode === 'single'
                ? singleFile == null
                : studies.length === 0 || studies.some((study) => study.files.length === 0))
            }
          >
            {loading ? 'Загрузка...' : 'Отправить на анализ'}
          </button>
          <label className="debug-toggle">
            <input
              type="checkbox"
              checked={debug}
              onChange={(event) => setDebug(event.target.checked)}
            />
            Показать debug-данные
          </label>
        </div>
      </form>

      {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}

      {predictionSingle && (
        <div className="result-block">
          <h2>Результат анализа:</h2>
          <div className="results-grid">
            <div className="result-card">
              <div className="result-item">
                <strong>Файл:</strong> {predictionSingle.filename ?? '—'}
              </div>
              <div className="result-item">
                <strong>Prediction:</strong> {String(predictionSingle.prediction ?? '—')}
              </div>
              <div className="result-item">
                <strong>Confidence:</strong>{' '}
                {predictionSingle.confidence != null ? predictionSingle.confidence.toFixed(4) : '—'}
              </div>
              <div className="result-item">
                <strong>Image:</strong>
                <img
                  className="result-image"
                  src={predictionSingle.image_base64}
                  alt="Result"
                />
              </div>
              {predictionSingle.debug && (
                <>
                  <div className="result-item">
                    <strong>Processed image:</strong>
                    <img
                      className="result-image"
                      src={predictionSingle.debug.processed_image}
                      alt="Processed"
                    />
                  </div>
                  {predictionSingle.debug.hog_image && (
                    <div className="result-item">
                      <strong>HOG image:</strong>
                      <img
                        className="result-image"
                        src={predictionSingle.debug.hog_image}
                        alt="HOG"
                      />
                    </div>
                  )}
                  <div className="result-item">
                    <strong>HOG vector:</strong>
                    <pre className="hog-output">{JSON.stringify(predictionSingle.debug.hog, null, 2)}</pre>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {predictionMultiple && (
        <div className="result-block">
          <h2>Результат анализа:</h2>
          <div className="results-grid">
            {predictionMultiple.map((item) => (
              <div className="result-card" key={item.study_id}>
                <div className="result-item">
                  <strong>Study:</strong> {studies.find((study) => study.id === item.study_id)?.name ?? item.study_id}
                </div>
                <div className="result-item">
                  <strong>Файлы:</strong> {item.filenames.join(', ') || '—'}
                </div>
                <div className="result-item">
                  <strong>Prediction:</strong> {String(item.prediction ?? '—')}
                </div>
                <div className="result-item">
                  <strong>Confidence:</strong>{' '}
                  {item.confidence != null ? item.confidence.toFixed(4) : '—'}
                </div>
                {item.debug && (
                  <>
                    <div className="result-item">
                      <strong>Processed image:</strong>
                      <img
                        className="result-image"
                        src={item.debug.processed_image}
                        alt="Processed"
                      />
                    </div>
                    {item.debug.hog_image && (
                      <div className="result-item">
                        <strong>HOG image:</strong>
                        <img
                          className="result-image"
                          src={item.debug.hog_image}
                          alt="HOG"
                        />
                      </div>
                    )}
                    <div className="result-item">
                      <strong>HOG vector:</strong>
                      <pre className="hog-output">{JSON.stringify(item.debug.hog, null, 2)}</pre>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
