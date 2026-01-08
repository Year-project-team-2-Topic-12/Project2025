import type { FormEvent } from 'react';

type AuthFormProps = {
  authToken: string | null;
  authUser: string | null;
  authLoading: boolean;
  authError: string | null;
  loginUsername: string;
  loginPassword: string;
  onLogin: (event: FormEvent<HTMLFormElement>) => void;
  onLogout: () => void;
  onLoginUsernameChange: (value: string) => void;
  onLoginPasswordChange: (value: string) => void;
};

export function AuthForm({
  authToken,
  authUser,
  authLoading,
  authError,
  loginUsername,
  loginPassword,
  onLogin,
  onLogout,
  onLoginUsernameChange,
  onLoginPasswordChange,
}: AuthFormProps) {
  return (
    <section className="auth-card">
      <div className="auth-header">
        <h2>Авторизация</h2>
        {authToken && (
          <button type="button" className="ghost-button" onClick={onLogout}>
            Выйти
          </button>
        )}
      </div>
      {authToken ? (
        <div className="auth-status">
          Вы вошли как <strong>{authUser ?? 'пользователь'}</strong>.
        </div>
      ) : (
        <form onSubmit={onLogin} className="auth-form">
          <label>
            Логин
            <input
              value={loginUsername}
              onChange={(event) => onLoginUsernameChange(event.target.value)}
              autoComplete="username"
            />
          </label>
          <label>
            Пароль
            <input
              type="password"
              value={loginPassword}
              onChange={(event) => onLoginPasswordChange(event.target.value)}
              autoComplete="current-password"
            />
          </label>
          <button type="submit" disabled={authLoading || !loginUsername || !loginPassword}>
            {authLoading ? 'Вход...' : 'Войти'}
          </button>
        </form>
      )}
      {authError && <div className="auth-error">{authError}</div>}
    </section>
  );
}
