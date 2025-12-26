import type { FormEvent } from 'react';
import type { ForwardImageResponse, PredictionResponse } from '../../client';
import type { StudyInput } from '../../types';
import { SingleUploadForm } from '../SingleUploadForm';
import { StudyUploadForm } from '../StudyUploadForm';

type PredictionsTabProps = {
  singleFile: File | null;
  loading: boolean;
  debug: boolean;
  onSingleSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onSingleFileChange: (file: File | null) => void;
  onDebugChange: (checked: boolean) => void;
  studies: StudyInput[];
  onStudySubmit: (event: FormEvent<HTMLFormElement>) => void;
  onStudyNameChange: (index: number, name: string) => void;
  onStudyFilesChange: (index: number, files: File[]) => void;
  onAddStudy: () => void;
  onRemoveStudy: (index: number) => void;
  error: string | null;
  predictionSingle: ForwardImageResponse | null;
  predictionMultiple: PredictionResponse[] | null;
};

export function PredictionsTab({
  singleFile,
  loading,
  debug,
  onSingleSubmit,
  onSingleFileChange,
  onDebugChange,
  studies,
  onStudySubmit,
  onStudyNameChange,
  onStudyFilesChange,
  onAddStudy,
  onRemoveStudy,
  error,
  predictionSingle,
  predictionMultiple,
}: PredictionsTabProps) {
  return (
    <>
      <SingleUploadForm
        singleFile={singleFile}
        loading={loading}
        debug={debug}
        onSubmit={onSingleSubmit}
        onFileChange={onSingleFileChange}
        onDebugChange={onDebugChange}
      />

      <StudyUploadForm
        studies={studies}
        loading={loading}
        debug={debug}
        onSubmit={onStudySubmit}
        onStudyNameChange={onStudyNameChange}
        onStudyFilesChange={onStudyFilesChange}
        onAddStudy={onAddStudy}
        onRemoveStudy={onRemoveStudy}
        onDebugChange={onDebugChange}
      />

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
                  <strong>Study:</strong>{' '}
                  {studies.find((study) => study.id === item.study_id)?.name ?? item.study_id}
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
    </>
  );
}
