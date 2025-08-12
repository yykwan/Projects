//
//  ContentView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: TaskViewModel
    @State private var filterCategory = "All"
    @State private var filterPriority = "All"
    @State private var showStore = false
    @State private var showInventory = false
    @State private var showSettings = false
    @State private var showResetAlert = false
    @State private var showAchievements = false
    @State private var selectedTaskID: UUID? = nil
    @State private var showEditTask = false

    var body: some View {
        GeometryReader { geo in
            VStack(spacing: 0) {
                // Top Status Bar
                VStack(spacing: 4) {
                    HStack(spacing: 0) {
                        StatusPill(icon: "❤️", value: viewModel.hearts)
                        StatusPill(icon: "😊", value: viewModel.happiness)
                        StatusPill(icon: "💰", value: viewModel.coins)
                    }
                    .padding(.horizontal, 4)

                    HStack(spacing: 0) {
                        Button(action: { showStore = true }) {
                            ActionPill(title: "Store")
                        }

                        Button(action: { showInventory = true }) {
                            ActionPill(title: "Inventory")
                        }

                        Button(action: { showSettings = true }) {
                            ActionPill(title: "Settings")
                        }
                    }
                    .padding(.horizontal, 4)
                }
                .frame(height: 60)
                .padding(.top, 4)

                // Cat GIF Display
                AnimatedGIFView(gifName: "typing_cat")
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geo.size.width * 0.8)
                    .frame(height: geo.size.height * 0.25)
                    .padding(.vertical, 4)

                // Task List
                TaskListView(viewModel: viewModel, filterCategory: $filterCategory, filterPriority: $filterPriority)
                    .frame(height: geo.size.height * 0.4)

                // Task Input
                TaskInputView(
                    viewModel: viewModel,
                    filterCategory: $filterCategory,
                    filterPriority: $filterPriority
                )
                .frame(height: geo.size.height * 0.2)
            }
            .frame(width: geo.size.width)
            .sheet(isPresented: $showStore) {
                StoreView(viewModel: viewModel, isPresented: $showStore)
            }
            .sheet(isPresented: $showInventory) {
                InventoryView(viewModel: viewModel, isPresented: $showInventory)
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(isPresented: $showSettings)
            }
            .sheet(isPresented: $showAchievements) {
                AchievementsView(viewModel: viewModel)
            }
            .sheet(isPresented: $showEditTask) {
                if let id = selectedTaskID,
                   let taskIndex = viewModel.tasks.firstIndex(where: { $0.id == id }) {
                    TaskEditView(
                        taskViewModel: viewModel,
                        task: $viewModel.tasks[taskIndex],
                        onSave: { updatedTask in
                            viewModel.updateTask(updatedTask)
                        }
                    )
                } else {
                    Text("Task not found")
                }
            }
            .alert("Game Reset", isPresented: $showResetAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("You ran out of hearts! Game has been reset but your tasks were saved.")
            }
            .onReceive(NotificationCenter.default.publisher(for: .appDidReset)) { _ in
                showResetAlert = true
            }
            
        }
    }
}

struct StatusPill: View {
    let icon: String
    let value: Int
    
    var body: some View {
        Text("\(icon) \(value)")
            .font(.system(size: 14, weight: .medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .frame(maxWidth: .infinity)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
    }
}

struct ActionPill: View {
    let title: String
    
    var body: some View {
        Text(title)
            .font(.system(size: 14))
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .frame(maxWidth: .infinity)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
    }
}
