import type { FormEvent } from 'react';
import type { UserResponse } from '../../client';
import { CreateUserForm } from '../CreateUserForm';

type UsersTabProps = {
  username: string;
  password: string;
  createLoading: boolean;
  createError: string | null;
  createSuccess: string | null;
  onCreateSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  users: UserResponse[];
  usersLoading: boolean;
  usersError: string | null;
  onRefreshUsers: () => void;
  onDeleteUser: (username: string) => void;
};

export function UsersTab({
  username,
  password,
  createLoading,
  createError,
  createSuccess,
  onCreateSubmit,
  onUsernameChange,
  onPasswordChange,
  users,
  usersLoading,
  usersError,
  onRefreshUsers,
  onDeleteUser,
}: UsersTabProps) {
  return (
    <section className="tab-panel">
      <CreateUserForm
        username={username}
        password={password}
        loading={createLoading}
        error={createError}
        success={createSuccess}
        onSubmit={onCreateSubmit}
        onUsernameChange={onUsernameChange}
        onPasswordChange={onPasswordChange}
      />
      <div className="panel-header">
        <h2>Пользователи</h2>
        <button type="button" className="ghost-button" onClick={onRefreshUsers} disabled={usersLoading}>
          Обновить
        </button>
      </div>
      {usersError && <div className="auth-error">{usersError}</div>}
      {usersLoading ? (
        <div className="panel-empty">Загрузка...</div>
      ) : users.length === 0 ? (
        <div className="panel-empty">Пользователей нет.</div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Логин</th>
                <th>Роль</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.username}>
                  <td>{user.username}</td>
                  <td>{user.role}</td>
                  <td>
                    {user.username !== 'admin' && (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => onDeleteUser(user.username)}
                        disabled={usersLoading}
                      >
                        Удалить
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
