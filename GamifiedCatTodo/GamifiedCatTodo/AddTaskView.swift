//
//  AddTaskView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//


import SwiftUI

struct AddTaskView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var manager: TaskManager

    @State private var title = ""
    @State private var category = ""
    @State private var priority: TaskPriority = .medium
    @State private var dueDate = Date()

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Task Info")) {
                    TextField("Title", text: $title)
                    TextField("Category", text: $category)

                    Picker("Priority", selection: $priority) {
                        ForEach(TaskPriority.allCases, id: \.self) { level in
                            Text(level.rawValue).tag(level)
                        }
                    }

                    DatePicker("Due Date", selection: $dueDate, displayedComponents: .date)
                        .datePickerStyle(.compact)
                }

                Section {
                    Button("Add Task") {
                        manager.addTask(title: title, category: category, priority: priority, dueDate: dueDate)
                        dismiss()
                    }
                    .disabled(title.isEmpty)
                }
            }
            .navigationTitle("New Task")
        }
    }
}
