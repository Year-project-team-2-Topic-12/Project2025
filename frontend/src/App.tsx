import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import './App.css';
import {
  forwardForwardPost,
  forwardMultipleForwardMultiplePost,
  deleteHistoryHistoryDelete,
  getStatsStatsGet,
  deleteUserAuthUsersUsernameDelete,
  loginAuthLoginPost,
  listUsersAuthUsersGet,
  readHistoryHistoryGet,
  registerAuthRegisterPost,
} from './client';
import { client } from './client/client.gen';
import { AuthForm } from './components/AuthForm';
import { HistoryTab } from './components/tabs/HistoryTab';
import { UsersTab } from './components/tabs/UsersTab';
import { StatsTab } from './components/tabs/StatsTab';
import { PredictionsTab } from './components/tabs/PredictionsTab';
import { ANATOMY_OPTIONS } from './types';
import type { Anatomy, StudyInput } from './types';
import type {
  BodyForwardForwardPost,
  BodyForwardMultipleForwardMultiplePost,
  ForwardForwardPostData,
  ForwardImageResponse,
  ForwardMultipleForwardMultiplePostData,
  PredictionResponse,
  RequestLogEntry,
  StatsResponse,
  UserResponse,
} from './client';

type LoginResponse = {
  access_token?: string;
  token_type?: string;
};

const unwrapData = <T,>(response: T | { data: T }): T => {
  if (response && typeof response === 'object' && 'data' in response) {
    return (response as { data: T }).data;
  }
  return response as T;
};

const createStudyId = () => {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  if (typeof globalThis.crypto?.getRandomValues === 'function') {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }

  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0'));
  return [
    hex.slice(0, 4).join(''),
    hex.slice(4, 6).join(''),
    hex.slice(6, 8).join(''),
    hex.slice(8, 10).join(''),
    hex.slice(10, 16).join(''),
  ].join('-');
};

function App() {
  const [authToken, setAuthToken] = useState<string | null>(() => localStorage.getItem('authToken'));
  const [authUser, setAuthUser] = useState<string | null>(() => localStorage.getItem('authUser'));
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [loginUsername, setLoginUsername] = useState('admin');
  const [loginPassword, setLoginPassword] = useState('');
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [createUserLoading, setCreateUserLoading] = useState(false);
  const [createUserError, setCreateUserError] = useState<string | null>(null);
  const [createUserSuccess, setCreateUserSuccess] = useState<string | null>(null);

  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singleAnatomy, setSingleAnatomy] = useState<Anatomy>('XR_SHOULDER');
  const [studies, setStudies] = useState<StudyInput[]>(() => [
    { id: createStudyId(), name: 'Study 1', anatomy: 'XR_SHOULDER', files: [] },
  ]);
  const [debug, setDebug] = useState<boolean>(false);

  const [predictionSingle, setPredictionSingle] = useState<ForwardImageResponse | null>(null);
  const [predictionMultiple, setPredictionMultiple] = useState<PredictionResponse[] | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<'history' | 'users' | 'stats' | 'predictions'>('predictions');
  const [historyItems, setHistoryItems] = useState<RequestLogEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [statsData, setStatsData] = useState<StatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [users, setUsers] = useState<UserResponse[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [usersError, setUsersError] = useState<string | null>(null);

  const isAdmin = authUser === 'admin';

  useEffect(() => {
    if (authToken) {
      localStorage.setItem('authToken', authToken);
    } else {
      localStorage.removeItem('authToken');
    }
    if (authUser) {
      localStorage.setItem('authUser', authUser);
    } else {
      localStorage.removeItem('authUser');
    }

    client.setConfig({
      headers: {
        Authorization: authToken ? `Bearer ${authToken}` : null,
      },
    });
  }, [authToken, authUser]);

  useEffect(() => {
    if (!isAdmin && activeTab === 'users') {
      setActiveTab('history');
    }
  }, [activeTab, isAdmin]);

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthLoading(true);
    setAuthError(null);

    try {
      const response = unwrapData<LoginResponse>(await loginAuthLoginPost({
        body: {
          username: loginUsername,
          password: loginPassword,
        },
        responseStyle: 'data',
        throwOnError: true,
      }));

      const token = response?.access_token;
      if (!token) {
        throw new Error('Не удалось получить токен');
      }

      setAuthToken(token);
      setAuthUser(loginUsername);
      setLoginPassword('');
    } catch (err) {
      if (err instanceof Error) {
        setAuthError(err.message);
      } else {
        setAuthError('Ошибка авторизации');
      }
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setAuthToken(null);
    setAuthUser(null);
    setPredictionSingle(null);
    setPredictionMultiple(null);
    setError(null);
  };

  const handleCreateUser = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!authToken) {
      setCreateUserError('Сначала выполните вход');
      return;
    }

    setCreateUserLoading(true);
    setCreateUserError(null);
    setCreateUserSuccess(null);

    try {
      const data = unwrapData<{ username?: string }>(await registerAuthRegisterPost({
        body: {
          username: newUsername,
          password: newPassword,
        },
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        responseStyle: 'data',
        throwOnError: true,
      }));
      const createdUsername =
        data && typeof data === 'object' && 'username' in data ? (data as { username?: string }).username : undefined;
      setCreateUserSuccess(`Пользователь ${createdUsername ?? newUsername} создан`);
      setNewUsername('');
      setNewPassword('');
      if (isAdmin) {
        await fetchUsers();
      }
    } catch (err) {
      if (err instanceof Error) {
        setCreateUserError(err.message);
      } else {
        setCreateUserError('Ошибка создания пользователя');
      }
    } finally {
      setCreateUserLoading(false);
    }
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const data = unwrapData<RequestLogEntry[]>(await readHistoryHistoryGet({
        responseStyle: 'data',
        throwOnError: true,
      }));
      setHistoryItems(data ?? []);
    } catch (err) {
      if (err instanceof Error) {
        setHistoryError(err.message);
      } else {
        setHistoryError('Ошибка загрузки истории');
      }
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleDeleteHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      await deleteHistoryHistoryDelete({
        responseStyle: 'data',
        throwOnError: true,
      });
      await fetchHistory();
    } catch (err) {
      if (err instanceof Error) {
        setHistoryError(err.message);
      } else {
        setHistoryError('Ошибка удаления истории');
      }
    } finally {
      setHistoryLoading(false);
    }
  };

  const fetchStats = async () => {
    setStatsLoading(true);
    setStatsError(null);
    try {
      const data = unwrapData<StatsResponse>(await getStatsStatsGet({
        responseStyle: 'data',
        throwOnError: true,
      }));
      setStatsData(data);
    } catch (err) {
      if (err instanceof Error) {
        setStatsError(err.message);
      } else {
        setStatsError('Ошибка загрузки статистики');
      }
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchUsers = async () => {
    setUsersLoading(true);
    setUsersError(null);
    try {
      const data = unwrapData<UserResponse[]>(await listUsersAuthUsersGet({
        responseStyle: 'data',
        throwOnError: true,
      }));
      setUsers(data ?? []);
    } catch (err) {
      if (err instanceof Error) {
        setUsersError(err.message);
      } else {
        setUsersError('Ошибка загрузки пользователей');
      }
    } finally {
      setUsersLoading(false);
    }
  };

  const handleDeleteUser = async (username: string) => {
    setUsersLoading(true);
    setUsersError(null);
    try {
      await deleteUserAuthUsersUsernameDelete({
        path: { username },
        responseStyle: 'data',
        throwOnError: true,
      });
      await fetchUsers();
    } catch (err) {
      if (err instanceof Error) {
        setUsersError(err.message);
      } else {
        setUsersError('Ошибка удаления пользователя');
      }
    } finally {
      setUsersLoading(false);
    }
  };

  useEffect(() => {
    if (!authToken) {
      return;
    }
    if (activeTab === 'history') {
      fetchHistory();
    }
    if (activeTab === 'stats') {
      fetchStats();
    }
    if (activeTab === 'users' && isAdmin) {
      fetchUsers();
    }
  }, [activeTab, authToken, isAdmin]);

  const updateStudyFiles = (index: number, files: File[]) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, files } : study)));
    setPredictionSingle(null);
    setPredictionMultiple(null);
    setError(null);
  };

  const updateStudyName = (index: number, name: string) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, name } : study)));
  };

  const updateStudyAnatomy = (index: number, anatomy: Anatomy) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, anatomy } : study)));
    setPredictionMultiple(null);
    setError(null);
  };

  const addStudy = () => {
    setStudies((prev) => [
      ...prev,
      { id: createStudyId(), name: `Study ${prev.length + 1}`, anatomy: 'XR_SHOULDER', files: [] },
    ]);
  };

  const removeStudy = (index: number) => {
    setStudies((prev) => prev.filter((_, i) => i !== index));
    setPredictionSingle(null);
    setPredictionMultiple(null);
  };

  const handleSingleFileChange = (file: File | null) => {
    setSingleFile(file);
    setPredictionSingle(null);
    setPredictionMultiple(null);
    setError(null);
  };

  const handleSingleAnatomyChange = (anatomy: Anatomy) => {
    setSingleAnatomy(anatomy);
    setPredictionSingle(null);
    setError(null);
  };

  const handleSingleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!singleFile) {
      alert("Пожалуйста, выберите файл!");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const authHeaders = authToken ? { Authorization: `Bearer ${authToken}` } : {};

      const body: BodyForwardForwardPost = { image: singleFile as File };
      const headers = {
        'X-Anatomy': singleAnatomy,
        'X-Debug': debug,
        ...authHeaders,
      } as ForwardForwardPostData['headers'] & { Authorization?: string };
      const data = unwrapData<ForwardImageResponse>(await forwardForwardPost({
        body,
        headers,
        responseStyle: 'data',
        throwOnError: true,
      }));
      setPredictionSingle(data);
      setPredictionMultiple(null);

    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        console.log(err);
        setError(`ошибка: ${String(err)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStudySubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (studies.length === 0 || studies.some((study) => study.files.length === 0)) {
      alert("Пожалуйста, добавьте хотя бы один файл в каждую группу!");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const authHeaders = authToken ? { Authorization: `Bearer ${authToken}` } : {};

      const files: File[] = [];
      const studyIds: string[] = [];
      const anatomies: Anatomy[] = [];
      studies.forEach((study) => {
        study.files.forEach((file) => {
          files.push(file);
          studyIds.push(study.id);
          anatomies.push(study.anatomy);
        });
      });

      const body: BodyForwardMultipleForwardMultiplePost = { images: files };
      const headers = {
        'X-Study-Ids': studyIds.join(','),
        'X-Anatomies': anatomies.join(','),
        'X-Debug': debug,
        ...authHeaders,
      } as ForwardMultipleForwardMultiplePostData['headers'] & { Authorization?: string };
      const data = unwrapData<PredictionResponse[]>(await forwardMultipleForwardMultiplePost({
        body,
        headers,
        responseStyle: 'data',
        throwOnError: true,
      }));
      setPredictionMultiple(data);
      setPredictionSingle(null);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        console.log(err);
        setError(`ошибка: ${String(err)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="app-hero">
        <div className="app-hero-copy">
          <span className="app-eyebrow">MURA / MLflow</span>
          <h1>MURA X-ray Analysis</h1>
        </div>
        <div className="app-hero-visual" aria-hidden="true">
          <div className="xray-panel xray-panel-primary" />
          <div className="xray-panel xray-panel-secondary" />
        </div>
      </header>

      <AuthForm
        authToken={authToken}
        authUser={authUser}
        authLoading={authLoading}
        authError={authError}
        loginUsername={loginUsername}
        loginPassword={loginPassword}
        onLogin={handleLogin}
        onLogout={handleLogout}
        onLoginUsernameChange={setLoginUsername}
        onLoginPasswordChange={setLoginPassword}
      />

      {authToken ? (
        <>
          <div className="tabs">
            {(isAdmin
              ? [
                { id: 'history', label: 'История' },
                { id: 'users', label: 'Пользователи' },
                { id: 'stats', label: 'Статистика' },
                { id: 'predictions', label: 'Предсказания' },
              ]
              : [
                { id: 'history', label: 'История' },
                { id: 'stats', label: 'Статистика' },
                { id: 'predictions', label: 'Предсказания' },
              ]
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === 'history' && (
            <HistoryTab
              items={historyItems}
              loading={historyLoading}
              error={historyError}
              isAdmin={isAdmin}
              onRefresh={fetchHistory}
              onDelete={handleDeleteHistory}
            />
          )}

          {activeTab === 'users' && isAdmin && (
            <UsersTab
              username={newUsername}
              password={newPassword}
              createLoading={createUserLoading}
              createError={createUserError}
              createSuccess={createUserSuccess}
              onCreateSubmit={handleCreateUser}
              onUsernameChange={setNewUsername}
              onPasswordChange={setNewPassword}
              users={users}
              usersLoading={usersLoading}
              usersError={usersError}
              onRefreshUsers={fetchUsers}
              onDeleteUser={handleDeleteUser}
            />
          )}

          {activeTab === 'stats' && (
            <StatsTab
              data={statsData}
              loading={statsLoading}
              error={statsError}
              onRefresh={fetchStats}
            />
          )}

          {activeTab === 'predictions' && (
            <PredictionsTab
              singleFile={singleFile}
              singleAnatomy={singleAnatomy}
              anatomyOptions={ANATOMY_OPTIONS}
              loading={loading}
              debug={debug}
              onSingleSubmit={handleSingleSubmit}
              onSingleFileChange={handleSingleFileChange}
              onSingleAnatomyChange={handleSingleAnatomyChange}
              onDebugChange={setDebug}
              studies={studies}
              onStudySubmit={handleStudySubmit}
              onStudyNameChange={updateStudyName}
              onStudyAnatomyChange={updateStudyAnatomy}
              onStudyFilesChange={updateStudyFiles}
              onAddStudy={addStudy}
              onRemoveStudy={removeStudy}
              error={error}
              predictionSingle={predictionSingle}
              predictionMultiple={predictionMultiple}
            />
          )}
        </>
      ) : (
        <div className="auth-hint">
          Войдите, чтобы отправлять изображения на анализ.
        </div>
      )}
    </div>
  );
}

export default App;
