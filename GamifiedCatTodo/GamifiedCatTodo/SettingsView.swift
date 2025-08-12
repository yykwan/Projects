//
//  SettingsView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 3/5/2025.
//
import SwiftUI

struct SettingsView: View {
    @Binding var isPresented: Bool
    @AppStorage("isDarkMode") private var isDarkMode = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("App Instructions")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("How to use this app:")
                            .font(.headline)
                        Text("1. Add tasks with due dates")
                        Text("2. Complete tasks to earn coins")
                        Text("3. Buy fish in the Store")
                        Text("4. Feed fish to increase happiness")
                        Text("5. Avoid overdue tasks to keep hearts")
                        Text("6. When lose all hearts, app resets!")
                    }
                    .padding(.vertical, 8)
                }
                
                Section(header: Text("Appearance")) {
                    Toggle(isOn: $isDarkMode) {
                        HStack {
                            Image(systemName: isDarkMode ? "moon.fill" : "sun.max.fill")
                            Text("Dark Mode")
                        }
                    }
                    .toggleStyle(SwitchToggleStyle(tint: .blue))
                    .onChange(of: isDarkMode, initial: false) { _,_  in
                        updateAppearance()
                    }
                }
                
                Section(header: Text("About")) {
                    Text("Version 1.0")
                    Text("Created by 🐱")
                }
            }
            .navigationTitle("Settings")
            .navigationBarItems(trailing: Button("Done") {
                isPresented = false
            })
            .preferredColorScheme(isDarkMode ? .dark : .light)
        }
    }
    
    private func updateAppearance() {
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            windowScene.windows.first?.overrideUserInterfaceStyle = isDarkMode ? .dark : .light
        }
    }
}
