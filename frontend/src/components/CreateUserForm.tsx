import type { FormEvent } from 'react';

type CreateUserFormProps = {
  username: string;
  password: string;
  loading: boolean;
  error: string | null;
  success: string | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
};

export function CreateUserForm({
  username,
  password,
  loading,
  error,
  success,
  onSubmit,
  onUsernameChange,
  onPasswordChange,
}: CreateUserFormProps) {
  return (
    <section className="auth-card">
      <h2>Создать пользователя</h2>
      <form onSubmit={onSubmit} className="auth-form">
        <label>
          Логин
          <input
            value={username}
            onChange={(event) => onUsernameChange(event.target.value)}
            autoComplete="off"
          />
        </label>
        <label>
          Пароль
          <input
            type="password"
            value={password}
            onChange={(event) => onPasswordChange(event.target.value)}
            autoComplete="new-password"
          />
        </label>
        <button type="submit" disabled={loading || !username || !password}>
          {loading ? 'Создание...' : 'Создать'}
        </button>
      </form>
      {error && <div className="auth-error">{error}</div>}
      {success && <div className="auth-success">{success}</div>}
    </section>
  );
}
