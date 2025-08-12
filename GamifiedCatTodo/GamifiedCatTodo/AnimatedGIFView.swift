//
//  AnimatedGIFView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI
import WebKit

struct AnimatedGIFView: UIViewRepresentable {
    let gifName: String
    
    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        guard let url = Bundle.main.url(forResource: gifName, withExtension: "gif") else { return }
        
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
        <style>
        body { margin: 0; padding: 0; background: transparent; }
        .gif-container { display: flex; justify-content: center; align-items: center; height: 100vh; }
        img { max-width: 100%; max-height: 100%; object-fit: contain; }
        </style>
        </head>
        <body>
        <div class="gif-container">
        <img src="\(url.lastPathComponent)">
        </div>
        </body>
        </html>
        """
        
        uiView.loadHTMLString(html, baseURL: url.deletingLastPathComponent())
    }
}
