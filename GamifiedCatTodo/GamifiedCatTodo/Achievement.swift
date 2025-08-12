//
//  Achievement.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 7/5/2025.
//


import Foundation

struct Achievement: Identifiable, Codable {
    let id: String
    let title: String
    let description: String
    let dateEarned: Date
    var isUnlocked: Bool = true
    
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: dateEarned)
    }
}

extension Achievement {
    static let sampleAchievements = [
        Achievement(id: "week_survivor", title: "Week Survivor", description: "Survived 7 days without reset", dateEarned: Date()),
        Achievement(id: "three_day_streak", title: "Three Day Streak", description: "Logged in for 3 consecutive days", dateEarned: Date())
    ]
}
