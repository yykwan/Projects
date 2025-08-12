//
//  TaskInputView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI

struct TaskInputView: View {
    @ObservedObject var viewModel: TaskViewModel
    @Binding var filterCategory: String
    @Binding var filterPriority: String

    @State private var title = ""
    @State private var category = "Work"
    @State private var priority = "Medium"
    @State private var dueDate = Date()

    var body: some View {
        VStack(spacing: 12) {
            TextField("Enter task title", text: $title)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal, 16)

            HStack(spacing: 8) {
                Picker("Category", selection: $category) {
                    ForEach(["Work", "Personal", "Study", "Chores"], id: \.self) {
                        Text($0)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)

                Picker("Priority", selection: $priority) {
                    ForEach(["Low", "Medium", "High"], id: \.self) {
                        Text($0)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 16)

            HStack(spacing: 8) {
                DatePicker("", selection: $dueDate, displayedComponents: .date)
                    .labelsHidden()
                    .frame(maxWidth: .infinity)

                DatePicker("", selection: $dueDate, displayedComponents: .hourAndMinute)
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 16)

            Button(action: addTask) {
                Text("Add Task")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding(.horizontal, 16)
            .disabled(title.isEmpty)
        }
        .padding(.vertical, 12)
    }

    private func addTask() {
        let newTask = Task(
            title: title,
            category: category,
            priority: TaskPriority(rawValue: priority) ?? .medium,
            dueDate: dueDate
        )
        viewModel.addTask(newTask)
        resetForm()
    }

    private func resetForm() {
        title = ""
        category = "Work"
        priority = "Medium"
        dueDate = Date()
    }
}
