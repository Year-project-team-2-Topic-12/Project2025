import type { FormEvent } from 'react';
import type { Anatomy } from '../types';

type SingleUploadFormProps = {
  singleFile: File | null;
  anatomy: Anatomy;
  anatomyOptions: readonly Anatomy[];
  loading: boolean;
  debug: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onFileChange: (file: File | null) => void;
  onAnatomyChange: (anatomy: Anatomy) => void;
  onDebugChange: (checked: boolean) => void;
};

export function SingleUploadForm({
  singleFile,
  anatomy,
  anatomyOptions,
  loading,
  debug,
  onSubmit,
  onFileChange,
  onAnatomyChange,
  onDebugChange,
}: SingleUploadFormProps) {
  return (
    <form onSubmit={onSubmit} className="upload-form">
      <h2>Одна картинка</h2>
      <div className="single-card">
        <label className="file-input">
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
              onFileChange(file);
            }}
          />
          <span>Выбрать файл</span>
        </label>
        <div className="file-hint">
          {singleFile ? singleFile.name : 'Пока нет файла'}
        </div>
        <label className="field-label">
          Anatomy
          <select
            className="anatomy-select"
            value={anatomy}
            onChange={(event) => onAnatomyChange(event.target.value as Anatomy)}
          >
            {anatomyOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="form-actions">
        <button type="submit" disabled={loading || singleFile == null}>
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
