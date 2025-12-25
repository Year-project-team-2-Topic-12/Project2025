import type { RequestLogEntry } from '../../client';

type HistoryTabProps = {
  items: RequestLogEntry[];
  loading: boolean;
  error: string | null;
  isAdmin: boolean;
  onRefresh: () => void;
  onDelete: () => void;
};

export function HistoryTab({
  items,
  loading,
  error,
  isAdmin,
  onRefresh,
  onDelete,
}: HistoryTabProps) {
  return (
    <section className="tab-panel">
      <div className="panel-header">
        <h2>История запросов</h2>
        <div className="panel-actions">
          <button type="button" className="ghost-button" onClick={onRefresh} disabled={loading}>
            Обновить
          </button>
          {isAdmin && (
            <button type="button" className="ghost-button" onClick={onDelete} disabled={loading}>
              Удалить историю
            </button>
          )}
        </div>
      </div>
      {error && <div className="auth-error">{error}</div>}
      {loading ? (
        <div className="panel-empty">Загрузка...</div>
      ) : items.length === 0 ? (
        <div className="panel-empty">История пока пустая.</div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Время</th>
                <th>Длительность</th>
                <th>Статус</th>
                <th>Результат</th>
                <th>Детали</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>{item.timestamp}</td>
                  <td>{item.duration ?? '—'}</td>
                  <td>{item.status ?? '—'}</td>
                  <td>{item.result ?? '—'}</td>
                  <td>{item.input_meta ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
