import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import logger from './utils/logger';
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: "https://74e92ef112fbc3aed76dd4f0169c70f8@o4510520744673280.ingest.us.sentry.io/4510549672853504",
  release: "alert-aid-frontend@1.0.0",
  environment: "production",
  integrations: [
    // Sentry Logs Integration (New) - Captures console logs as Sentry Logs
    Sentry.consoleLoggingIntegration({ levels: ["log", "warn", "error", "info", "debug"] }),
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration(),
    Sentry.browserProfilingIntegration(),
    Sentry.feedbackIntegration({
      colorScheme: "system",
      isNameRequired: true,
      isEmailRequired: true,
    }),
  ],
  tracesSampleRate: 1.0,
  profilesSampleRate: 1.0,
  replaysSessionSampleRate: 1.0, // Capture 100% of sessions
  replaysOnErrorSampleRate: 1.0,
  tracePropagationTargets: ["localhost", /^https:\/\/yourserver\.io\/api/],
});

// Test metrics and logs
Sentry.captureMessage('User triggered test log', { level: 'info', tags: { log_source: 'sentry_test' } });
Sentry.metrics.count('button_click', 1);
Sentry.metrics.gauge('page_load_time', 150);
Sentry.metrics.distribution('response_time', 200);

// Force cache clear and unregister service workers
logger.log('🌟 Alert Aid - API Endpoints Fixed v2.0.0');
logger.log('📅 Build Date:', new Date().toLocaleString());

// Unregister any old service workers
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(function(registrations) {
    for(let registration of registrations) {
      registration.unregister();
      logger.log('🧹 Unregistered old service worker');
    }
  });
}

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
