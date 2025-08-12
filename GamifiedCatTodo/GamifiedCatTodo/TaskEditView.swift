//
//  TaskEditView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 7/5/2025.
//
import SwiftUI
import UserNotifications

struct TaskEditView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var taskViewModel: TaskViewModel
    @Binding var task: Task
    var onSave: (Task) -> Void

    @State private var customCategory: String = ""
    @State private var dueDateTime: Date = Date()
    @State private var notificationDateTime: Date = Date()

    let taskCategories = ["Work", "Personal", "Study", "Chores", "Custom"]

    var body: some View {
        Form {
            Section(header: Text("Task Details")) {
                TextField("Title", text: $task.title)

                // Category Picker
                Picker("Category", selection: $task.category) {
                    ForEach(taskCategories, id: \.self) { category in
                        Text(category)
                    }
                }
                .pickerStyle(MenuPickerStyle())

                if task.category == "Custom" {
                    TextField("Enter Custom Category", text: $customCategory)
                }

                // Priority Picker
                Picker("Priority", selection: $task.priority) {
                    ForEach(TaskPriority.allCases, id: \.self) { priority in
                        Text(priority.rawValue.capitalized).tag(priority)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())

                // Due Date
                DatePicker("Due Date & Time", selection: $dueDateTime, displayedComponents: [.date, .hourAndMinute])

                // Reminder toggle and picker
                Toggle("Set Reminder", isOn: $task.wantsNotification)

                if task.wantsNotification {
                    DatePicker("Reminder Date & Time", selection: $notificationDateTime, displayedComponents: [.date, .hourAndMinute])
                }
            }

            Section {
                Button("Save") {
                    // Update custom category if needed
                    if task.category == "Custom", !customCategory.isEmpty {
                        task.category = customCategory
                    }

                    // Assign dates
                    task.dueDate = dueDateTime
                    task.notificationDate = task.wantsNotification ? notificationDateTime : nil

                    // Schedule notification
                    if task.wantsNotification {
                        scheduleNotification(for: task)
                    }

                    onSave(task)
                    dismiss()
                }
                .disabled(task.title.isEmpty)
            }
        }
        .navigationTitle("Edit Task")
        .onAppear {
            customCategory = task.category == "Custom" ? task.category : ""
            dueDateTime = task.dueDate ?? Date()
            notificationDateTime = task.notificationDate ?? Date()
        }
    }

    // MARK: - Notification Scheduling
    private func scheduleNotification(for task: Task) {
        guard let notificationDate = task.notificationDate else { return }

        let content = UNMutableNotificationContent()
        content.title = "Task Reminder"
        content.body = task.title
        content.sound = .default

        let triggerDate = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: notificationDate)
        let trigger = UNCalendarNotificationTrigger(dateMatching: triggerDate, repeats: false)

        let request = UNNotificationRequest(identifier: task.id.uuidString, content: content, trigger: trigger)

        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("❌ Notification error: \(error.localizedDescription)")
            }
        }
    }
}
