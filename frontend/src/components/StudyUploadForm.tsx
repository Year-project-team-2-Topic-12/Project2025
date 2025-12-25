import type { FormEvent } from 'react';
import type { StudyInput } from '../types';

type StudyUploadFormProps = {
  studies: StudyInput[];
  loading: boolean;
  debug: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onStudyNameChange: (index: number, name: string) => void;
  onStudyFilesChange: (index: number, files: File[]) => void;
  onAddStudy: () => void;
  onRemoveStudy: (index: number) => void;
  onDebugChange: (checked: boolean) => void;
};

export function StudyUploadForm({
  studies,
  loading,
  debug,
  onSubmit,
  onStudyNameChange,
  onStudyFilesChange,
  onAddStudy,
  onRemoveStudy,
  onDebugChange,
}: StudyUploadFormProps) {
  const isInvalid =
    studies.length === 0 || studies.some((study) => study.files.length === 0);

  return (
    <form onSubmit={onSubmit} className="upload-form">
      <h2>Исследование / несколько исследований</h2>
      <div className="study-grid">
        {studies.map((study, index) => (
          <div className="study-card" key={study.id}>
            <div className="study-header">
              <input
                className="study-name"
                value={study.name}
                onChange={(event) => onStudyNameChange(index, event.target.value)}
                placeholder={`Study ${index + 1}`}
              />
              {studies.length > 1 && (
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => onRemoveStudy(index)}
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
                  onStudyFilesChange(index, event.target.files ? Array.from(event.target.files) : [])
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
        <button type="button" className="add-card" onClick={onAddStudy}>
          + Добавить study
        </button>
      </div>

      <div className="form-actions">
        <button type="submit" disabled={loading || isInvalid}>
          {loading ? 'Загрузка...' : 'Отправить на анализ'}
        </button>
        <label className="debug-toggle">
          <input
            type="checkbox"
            checked={debug}
            onChange={(event) => onDebugChange(event.target.checked)}
          />
          Показать debug-данные
        </label>
      </div>
    </form>
  );
}
