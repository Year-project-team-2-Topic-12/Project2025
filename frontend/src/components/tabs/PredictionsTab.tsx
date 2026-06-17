import type { FormEvent } from 'react';
import type { ForwardImageResponse, PredictionResponse } from '../../client';
import type { Anatomy, StudyInput } from '../../types';
import { SingleUploadForm } from '../SingleUploadForm';
import { StudyUploadForm } from '../StudyUploadForm';

type PredictionsTabProps = {
  singleFile: File | null;
  singleAnatomy: Anatomy;
  anatomyOptions: readonly Anatomy[];
  loading: boolean;
  debug: boolean;
  onSingleSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onSingleFileChange: (file: File | null) => void;
  onSingleAnatomyChange: (anatomy: Anatomy) => void;
  onDebugChange: (checked: boolean) => void;
  studies: StudyInput[];
  onStudySubmit: (event: FormEvent<HTMLFormElement>) => void;
  onStudyNameChange: (index: number, name: string) => void;
  onStudyAnatomyChange: (index: number, anatomy: Anatomy) => void;
  onStudyFilesChange: (index: number, files: File[]) => void;
  onAddStudy: () => void;
  onRemoveStudy: (index: number) => void;
  error: string | null;
  predictionSingle: ForwardImageResponse | null;
  predictionMultiple: PredictionResponse[] | null;
};

export function PredictionsTab({
  singleFile,
  singleAnatomy,
  anatomyOptions,
  loading,
  debug,
  onSingleSubmit,
  onSingleFileChange,
  onSingleAnatomyChange,
  onDebugChange,
  studies,
  onStudySubmit,
  onStudyNameChange,
  onStudyAnatomyChange,
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
        anatomy={singleAnatomy}
        anatomyOptions={anatomyOptions}
        loading={loading}
        debug={debug}
        onSubmit={onSingleSubmit}
        onFileChange={onSingleFileChange}
        onAnatomyChange={onSingleAnatomyChange}
        onDebugChange={onDebugChange}
      />

      <StudyUploadForm
        studies={studies}
        anatomyOptions={anatomyOptions}
        loading={loading}
        debug={debug}
        onSubmit={onStudySubmit}
        onStudyNameChange={onStudyNameChange}
        onStudyAnatomyChange={onStudyAnatomyChange}
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
                <strong>Anatomy:</strong> {predictionSingle.anatomy ?? '—'}
              </div>
              <div className="result-item">
                <strong>Probability:</strong>{' '}
                {predictionSingle.probability != null ? predictionSingle.probability.toFixed(4) : '—'}
              </div>
              <div className="result-item">
                <strong>Confidence:</strong>{' '}
                {predictionSingle.confidence != null ? predictionSingle.confidence.toFixed(4) : '—'}
              </div>
              <div className="result-item">
                <strong>Threshold:</strong>{' '}
                {predictionSingle.threshold != null ? predictionSingle.threshold.toFixed(3) : '—'}
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
                  {predictionSingle.debug.processed_image && (
                    <div className="result-item">
                      <strong>Processed image:</strong>
                      <img
                        className="result-image"
                        src={predictionSingle.debug.processed_image}
                        alt="Processed"
                      />
                    </div>
                  )}
                  {predictionSingle.debug.image_predictions && (
                    <pre className="debug-output">
                      {JSON.stringify(predictionSingle.debug.image_predictions, null, 2)}
                    </pre>
                  )}
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
                  <strong>Anatomy:</strong> {item.anatomy ?? '—'}
                </div>
                <div className="result-item">
                  <strong>Images:</strong> {item.n_images ?? item.filenames.length}
                </div>
                <div className="result-item">
                  <strong>Probability:</strong>{' '}
                  {item.probability != null ? item.probability.toFixed(4) : '—'}
                </div>
                <div className="result-item">
                  <strong>Confidence:</strong>{' '}
                  {item.confidence != null ? item.confidence.toFixed(4) : '—'}
                </div>
                <div className="result-item">
                  <strong>Threshold:</strong>{' '}
                  {item.threshold != null ? item.threshold.toFixed(3) : '—'}
                </div>
                {item.debug && (
                  <>
                    {item.debug.processed_image && (
                      <div className="result-item">
                        <strong>Processed image:</strong>
                        <img
                          className="result-image"
                          src={item.debug.processed_image}
                          alt="Processed"
                        />
                      </div>
                    )}
                    {item.debug.image_predictions && (
                      <pre className="debug-output">
                        {JSON.stringify(item.debug.image_predictions, null, 2)}
                      </pre>
                    )}
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
