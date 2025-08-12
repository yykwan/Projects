//
//  UIImage+GIF.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//

import UIKit
import ImageIO

extension UIImage {
    static func gif(data: Data) -> UIImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }

        var images: [UIImage] = []
        var duration: Double = 0

        let count = CGImageSourceGetCount(source)
        for i in 0..<count {
            guard let cgImage = CGImageSourceCreateImageAtIndex(source, i, nil) else { continue }

            let frameDuration = UIImage.frameDuration(from: source, at: i)
            duration += frameDuration
            images.append(UIImage(cgImage: cgImage))
        }

        return UIImage.animatedImage(with: images, duration: duration)
    }

    private static func frameDuration(from source: CGImageSource, at index: Int) -> Double {
        let defaultFrameDuration = 0.1
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any],
              let gifDict = properties[kCGImagePropertyGIFDictionary] as? [CFString: Any],
              let delay = gifDict[kCGImagePropertyGIFUnclampedDelayTime] as? Double ?? gifDict[kCGImagePropertyGIFDelayTime] as? Double
        else {
            return defaultFrameDuration
        }

        return delay < 0.011 ? defaultFrameDuration : delay
    }
}
