# Frontend

This is the React frontend for the Multimodal Image Search application. It provides an intuitive UI for searching images by text, matching images to images, and finding captions for images.

## Setup

The frontend requires Node.js 20+.

`ash
cd frontend
npm ci
`

## Development

Run the Vite development server:

`ash
npm run dev
`

During development, Vite automatically proxies API requests (/search, /images, and /health) to the FastAPI backend running at http://localhost:8000.

## Production & Docker

For production, the frontend is compiled into static HTML/CSS/JS and served via Nginx.

`ash
npm run build
`

When deployed via Docker, the 
ginx.conf sets up a reverse proxy that automatically routes /search, /images, and /health to the ackend:8000 container. The frontend uses relative URLs, so VITE_API_URL can remain unset in .env.production.
