// JCP CHAT - Service Worker for PWA Support
const CACHE_VERSION = 'jcp-chat-v1';
const RUNTIME_CACHE = 'jcp-chat-runtime';
const IMAGE_CACHE = 'jcp-chat-images';

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/static/image.jpg',
  '/static/service-worker.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => {
      console.log('[ServiceWorker] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_VERSION && cacheName !== IMAGE_CACHE && cacheName !== RUNTIME_CACHE) {
            console.log('[ServiceWorker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - implement caching strategy
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Handle API calls - Network first, fall back to cache
  if (url.pathname === '/ask') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (!response || response.status !== 200) {
            return response;
          }
          const responseClone = response.clone();
          caches.open(RUNTIME_CACHE).then((cache) => {
            cache.put(request, responseClone);
          });
          return response;
        })
        .catch(() => {
          // Return cached response if network fails
          return caches.match(request).then((cachedResponse) => {
            return cachedResponse || new Response(
              JSON.stringify({ error: 'You are offline. Try again when connected.' }),
              { status: 503, headers: { 'Content-Type': 'application/json' } }
            );
          });
        })
    );
    return;
  }

  // Handle images - Cache first, fall back to network
  if (request.headers.get('accept').includes('image')) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        return (
          cachedResponse ||
          fetch(request).then((response) => {
            if (!response || response.status !== 200) {
              return response;
            }
            const responseClone = response.clone();
            caches.open(IMAGE_CACHE).then((cache) => {
              cache.put(request, responseClone);
            });
            return response;
          })
        );
      })
    );
    return;
  }

  // Handle HTML, CSS, JS - Stale while revalidate
  if (
    request.headers.get('accept').includes('text/html') ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.js')
  ) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        const fetchPromise = fetch(request).then((response) => {
          if (!response || response.status !== 200) {
            return response;
          }
          const responseClone = response.clone();
          caches.open(CACHE_VERSION).then((cache) => {
            cache.put(request, responseClone);
          });
          return response;
        });
        return cachedResponse || fetchPromise;
      })
    );
    return;
  }

  // Default network first
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (!response || response.status !== 200) {
          return response;
        }
        const responseClone = response.clone();
        caches.open(RUNTIME_CACHE).then((cache) => {
          cache.put(request, responseClone);
        });
        return response;
      })
      .catch(() => {
        return caches.match(request);
      })
  );
});

// Background sync for messages
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-messages') {
    event.waitUntil(
      // Retry failed messages when back online
      new Promise((resolve) => {
        console.log('[ServiceWorker] Syncing messages...');
        resolve();
      })
    );
  }
});

// Push notifications
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'JCP CHAT notification',
    icon: 'static/icon-192x192.png',
    badge: 'static/icon-192x192.png',
    tag: 'jcp-notification',
    requireInteraction: false
  };

  event.waitUntil(
    self.registration.showNotification('JCP CHAT', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === '/' && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow('/');
      }
    })
  );
});