const CACHE_NAME = 'asuna-v2';
const ASSETS_TO_CACHE = [
  // '/',
  // '/style_design_tokens.css',
  // '/style_page.css',
  // '/style_anime.css',
  // '/style_chat.css',
  // '/script_theme.js',
  // '/script_global.js',
  // '/script_user.js',
  // '/script_installer.js',
  // '/script_app.js',
  // '/script_bot.js',
  // '/script_chat.js',
  // '/script_anime.js',
  // '/script_page_handler.js',
  // '/script_pwa.js',
  // '/manifest.webmanifest',
  '/icons/icon-192x192.png',
  '/icons/icon-256x256.png',
  '/icons/icon-384x384.png',
  '/icons/icon-512x512.png',
  'https://i.ibb.co/jGGGYw4/image.webp',
  'https://i.ibb.co/KqYGB5t/image.webp',
  'https://i.ibb.co/WVyzpvz/image.webp',
  'https://i.ibb.co/8dbTJbM/image.webp',
  'https://i.ibb.co/9tbsVCB/image.webp',
  'https://cdn.jsdelivr.net/gh/hung1001/font-awesome-pro-v6@44659d9/css/all.min.css'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            return caches.delete(cache);
          }
        })
      );
      // NOTE: clients.claim() intentionally omitted.
      // Calling it mid-session wipes history.pushState entries (our back-guard trap).
      // The SW will take control on the next navigation naturally.
    })
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Exclude dynamic API calls, voice endpoints and status ping from service worker cache
  if (url.pathname.startsWith('/chat') ||
    url.pathname.startsWith('/login') ||
    url.pathname.startsWith('/signup') ||
    url.pathname.startsWith('/dl_data') ||
    url.pathname.startsWith('/voice') ||
    url.pathname.startsWith('/ping')) {
    return;
  }

  // Network First with Cache Fallback strategy
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response && response.status === 200) {
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return response;
      })
      .catch(() => {
        return caches.match(event.request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          if (event.request.headers.get('accept') && event.request.headers.get('accept').includes('text/html')) {
            return caches.match('/');
          }
        });
      })
  );
});
