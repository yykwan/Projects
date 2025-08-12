//
//  AchievementsView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 7/5/2025.
//


import SwiftUI

struct AchievementsView: View {
    @ObservedObject var viewModel: TaskViewModel
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Your Achievements (\(viewModel.achievements.count))")) {
                    if viewModel.achievements.isEmpty {
                        Text("No achievements yet. Keep playing to unlock some!")
                            .foregroundColor(.secondary)
                    } else {
                        ForEach(viewModel.achievements) { achievement in
                            AchievementRow(achievement: achievement)
                        }
                    }
                }
                
                Section(header: Text("Stats")) {
                    StatRow(label: "Highest Hearts", value: "\(viewModel.highestHeartsMaintained)")
                    StatRow(label: "Days Survived", value: "\(viewModel.daysSurvived)")
                    StatRow(label: "Last Reset", 
                           value: viewModel.lastResetDate.formatted(date: .abbreviated, time: .omitted))
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Achievements")
        }
    }
}

struct AchievementRow: View {
    let achievement: Achievement
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "star.fill")
                .foregroundColor(.yellow)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(achievement.title)
                    .font(.headline)
                Text(achievement.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text("Unlocked: \(achievement.formattedDate)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 8)
    }
}

struct StatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundColor(.secondary)
        }
    }
}