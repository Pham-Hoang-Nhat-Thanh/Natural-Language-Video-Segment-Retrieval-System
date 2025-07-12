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
        <title>Video Search System</title>
        <meta name="description" content="Natural language video search" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="antialiased">
        <header className="bg-gray-800 text-white p-4">
          <nav className="max-w-7xl mx-auto flex space-x-4">
            <Link href="/">Search</Link>
            <Link href="/videos">Videos</Link>
            <Link href="/admin">Admin</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  )
}
