//
//  GIFImageView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI
import ImageIO

struct GIFImageView: UIViewRepresentable {
    var gifName: String
    
    func makeUIView(context: Context) -> UIImageView {
        let imageView = UIImageView()
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        return imageView
    }

    func updateUIView(_ uiView: UIImageView, context: Context) {
        if let gifURL = Bundle.main.url(forResource: gifName, withExtension: "gif") {
            let gifData = try? Data(contentsOf: gifURL)
            let imageSource = CGImageSourceCreateWithData(gifData! as CFData, nil)
            let imageCount = CGImageSourceGetCount(imageSource!)
            
            var images = [UIImage]()
            for i in 0..<imageCount {
                if let cgImage = CGImageSourceCreateImageAtIndex(imageSource!, i, nil) {
                    images.append(UIImage(cgImage: cgImage))
                }
            }
            
            uiView.animationImages = images
            uiView.animationDuration = Double(imageCount) * 0.1
            uiView.animationRepeatCount = 0
            uiView.startAnimating()
        }
    }
}
