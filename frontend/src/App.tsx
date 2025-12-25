import { useEffect, useState, FormEvent } from 'react';
import './App.css';
import {
  forwardForwardPost,
  forwardMultipleForwardMultiplePost,
  loginAuthLoginPost,
  registerAuthRegisterPost,
} from './client';
import { client } from './client/client.gen';
import { AuthForm } from './components/AuthForm';
import { CreateUserForm } from './components/CreateUserForm';
import { SingleUploadForm } from './components/SingleUploadForm';
import { StudyUploadForm } from './components/StudyUploadForm';
import type { StudyInput } from './types';
import type {
  BodyForwardForwardPost,
  BodyForwardMultipleForwardMultiplePost,
  ForwardForwardPostData,
  ForwardImageResponse,
  ForwardMultipleForwardMultiplePostData,
  PredictionResponse,
} from './client';

type LoginResponse = {
  access_token?: string;
  token_type?: string;
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
  const [studies, setStudies] = useState<StudyInput[]>([
    { id: crypto.randomUUID(), name: 'Study 1', files: [] },
  ]);
  const [debug, setDebug] = useState<boolean>(false);

  const [predictionSingle, setPredictionSingle] = useState<ForwardImageResponse | null>(null);
  const [predictionMultiple, setPredictionMultiple] = useState<PredictionResponse[] | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthLoading(true);
    setAuthError(null);

    try {
      const response = (await loginAuthLoginPost({
        body: {
          username: loginUsername,
          password: loginPassword,
        },
        responseStyle: 'data',
        throwOnError: true,
      })) as LoginResponse;

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
      const data = await registerAuthRegisterPost({
        body: {
          username: newUsername,
          password: newPassword,
        },
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        responseStyle: 'data',
        throwOnError: true,
      });
      const createdUsername =
        data && typeof data === 'object' && 'username' in data ? (data as { username?: string }).username : undefined;
      setCreateUserSuccess(`Пользователь ${createdUsername ?? newUsername} создан`);
      setNewUsername('');
      setNewPassword('');
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

  const updateStudyFiles = (index: number, files: File[]) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, files } : study)));
    setPredictionSingle(null);
    setPredictionMultiple(null);
    setError(null);
  };

  const updateStudyName = (index: number, name: string) => {
    setStudies((prev) => prev.map((study, i) => (i === index ? { ...study, name } : study)));
  };

  const addStudy = () => {
    setStudies((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: `Study ${prev.length + 1}`, files: [] },
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
        'X-Debug': debug,
        ...authHeaders,
      } as ForwardForwardPostData['headers'] & { Authorization?: string };
      const data = await forwardForwardPost({
        body,
        headers,
        responseStyle: 'data',
        throwOnError: true,
      });
      setPredictionSingle(data);
      setPredictionMultiple(null);

    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        console.log(err);
        setError(`ошибка: ${err.message || err.detail || String(err)}`);
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
      studies.forEach((study) => {
        study.files.forEach((file) => {
          files.push(file);
          studyIds.push(study.id);
        });
      });

      const body: BodyForwardMultipleForwardMultiplePost = { images: files };
      const headers = {
        'X-Study-Ids': studyIds.join(','),
        'X-Debug': debug,
        ...authHeaders,
      } as ForwardMultipleForwardMultiplePostData['headers'] & { Authorization?: string };
      const data = await forwardMultipleForwardMultiplePost({
        body,
        headers,
        responseStyle: 'data',
        throwOnError: true,
      });
      setPredictionMultiple(data);
      setPredictionSingle(null);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        console.log(err);
        setError(`ошибка: ${err.message || err.detail || String(err)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>MURA ML Upload</h1>

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
          {authUser === 'admin' && (
            <CreateUserForm
              username={newUsername}
              password={newPassword}
              loading={createUserLoading}
              error={createUserError}
              success={createUserSuccess}
              onSubmit={handleCreateUser}
              onUsernameChange={setNewUsername}
              onPasswordChange={setNewPassword}
            />
          )}

          <SingleUploadForm
            singleFile={singleFile}
            loading={loading}
            debug={debug}
            onSubmit={handleSingleSubmit}
            onFileChange={handleSingleFileChange}
            onDebugChange={setDebug}
          />

          <StudyUploadForm
            studies={studies}
            loading={loading}
            debug={debug}
            onSubmit={handleStudySubmit}
            onStudyNameChange={updateStudyName}
            onStudyFilesChange={updateStudyFiles}
            onAddStudy={addStudy}
            onRemoveStudy={removeStudy}
            onDebugChange={setDebug}
          />
        </>
      ) : (
        <div className="auth-hint">
          Войдите, чтобы отправлять изображения на анализ.
        </div>
      )}

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
                  <strong>Study:</strong> {studies.find((study) => study.id === item.study_id)?.name ?? item.study_id}
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
    </div>
  );
}

export default App;
