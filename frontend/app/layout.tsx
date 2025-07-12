import React from 'react'
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
        {children}
      </body>
    </html>
  )
}
