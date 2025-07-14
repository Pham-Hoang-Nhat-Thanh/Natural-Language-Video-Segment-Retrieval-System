import React from 'react'
import Link from 'next/link'
import './globals.css'

export default function RootLayout({
  children,
}: {
  children: any
}) {
  return (
    <html lang="en">
      <head>
        <title>Video Segment Retrieval System</title>
        <meta name="description" content="Natural language video segment search and retrieval" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="antialiased bg-gray-50">
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <h1 className="text-xl font-bold text-gray-900">
                    Video Retrieval
                  </h1>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link 
                    href="/" 
                    className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-blue-600 border-b-2 border-transparent hover:border-blue-300 transition-colors"
                  >
                    Search
                  </Link>
                  <Link 
                    href="/manage" 
                    className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent hover:border-gray-300 transition-colors"
                  >
                    Manage Videos
                  </Link>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="text-sm text-gray-500">
                  Powered by CLIP + FastAPI
                </div>
              </div>
            </div>
          </div>
        </nav>
        {children}
      </body>
    </html>
  )
}
