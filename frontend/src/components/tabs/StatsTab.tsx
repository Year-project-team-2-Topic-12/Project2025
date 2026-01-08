import type { StatsResponse } from '../../client';

type StatsTabProps = {
  data: StatsResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
};

export function StatsTab({
  data,
  loading,
  error,
  onRefresh,
}: StatsTabProps) {
  return (
    <section className="tab-panel">
      <div className="panel-header">
        <h2>Статистика</h2>
        <button type="button" className="ghost-button" onClick={onRefresh} disabled={loading}>
          Обновить
        </button>
      </div>
      {error && <div className="auth-error">{error}</div>}
      {loading ? (
        <div className="panel-empty">Загрузка...</div>
      ) : data ? (
        <div className="stats-grid">
          <div className="stats-card">
            <h3>Время обработки</h3>
            <p>Среднее: {data.processing_time.mean_ms.toFixed(2)} ms</p>
            <p>P50: {data.processing_time.p50_ms.toFixed(2)} ms</p>
            <p>P95: {data.processing_time.p95_ms.toFixed(2)} ms</p>
            <p>P99: {data.processing_time.p99_ms.toFixed(2)} ms</p>
          </div>
          <div className="stats-card">
            <h3>Размеры изображений</h3>
            <p>Средняя ширина: {data.image_stats.mean_width.toFixed(2)}</p>
            <p>Средняя высота: {data.image_stats.mean_height.toFixed(2)}</p>
            <p>Кол-во: {data.image_stats.count}</p>
          </div>
        </div>
      ) : (
        <div className="panel-empty">Нет данных.</div>
      )}
    </section>
  );
}
