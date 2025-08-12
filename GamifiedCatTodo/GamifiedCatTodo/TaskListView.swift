//
//  TaskListView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI

// MARK: - TaskListView

struct TaskListView: View {
    @ObservedObject var viewModel: TaskViewModel
    @Binding var filterCategory: String
    @Binding var filterPriority: String

    @State private var editingTask: Task?
    @State private var showEditView = false

    var body: some View {
        List {
            ForEach(sortedTasks) { task in
                if shouldShow(task: task) {
                    TaskRowView(task: binding(for: task), viewModel: viewModel)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            viewModel.toggleCompletion(for: task)
                        }
                        .onLongPressGesture {
                            editingTask = task
                            showEditView = true
                        }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                viewModel.deleteTask(task)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }

                            Button {
                                viewModel.togglePinning(for: task)
                            } label: {
                                Label(task.isPinned ? "Unpin" : "Pin", systemImage: task.isPinned ? "pin.slash" : "pin")
                            }
                            .tint(.yellow)
                        }
                        .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                }
            }
        }
        .listStyle(.plain)
        .sheet(item: $editingTask) { task in
            if let index = viewModel.tasks.firstIndex(where: { $0.id == task.id }) {
                TaskEditView(
                    taskViewModel: viewModel,
                    task: $viewModel.tasks[index],
                    onSave: { updatedTask in
                        viewModel.updateTask(updatedTask)
                        showEditView = false
                    }
                )
            }
        }
    }

    // MARK: - Helpers

    private func binding(for task: Task) -> Binding<Task> {
        guard let index = viewModel.tasks.firstIndex(where: { $0.id == task.id }) else {
            fatalError("Task not found")
        }
        return $viewModel.tasks[index]
    }

    private var sortedTasks: [Task] {
        viewModel.tasks
            .filter(shouldShow(task:))
            .sorted { t1, t2 in
                if t1.isPinned != t2.isPinned {
                    return t1.isPinned && !t2.isPinned  // Pinned first
                } else {
                    return (t1.dueDate ?? Date.distantFuture) < (t2.dueDate ?? Date.distantFuture)
                }
            }
    }

    private func shouldShow(task: Task) -> Bool {
        (filterCategory == "All" || task.category == filterCategory) &&
        (filterPriority == "All" || task.priority.rawValue == filterPriority)
    }
}

// MARK: - TaskRowView

struct TaskRowView: View {
    @Binding var task: Task
    @ObservedObject var viewModel: TaskViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 12) {
                // Completion circle
                Button(action: {
                    viewModel.toggleCompletion(for: task)
                }) {
                    Image(systemName: task.isCompleted ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundColor(task.isCompleted ? .green : .gray)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(task.title)
                        .strikethrough(task.isCompleted)
                        .foregroundColor(task.isCompleted ? .gray : .primary)

                    HStack(spacing: 8) {
                        Text(task.category)
                            .font(.caption)
                            .padding(4)
                            .background(Color.gray.opacity(0.2))
                            .cornerRadius(4)

                        Text(task.priority.rawValue)
                            .font(.caption)
                            .padding(4)
                            .background(priorityColor(task.priority))
                            .cornerRadius(4)

                        if let dueDate = task.dueDate {
                            Text(dueDate, style: .date)
                                .font(.caption2)
                            Text(dueDate, style: .time)
                                .font(.caption2)
                        }
                    }
                }

                Spacer()

                if task.isPinned {
                    Image(systemName: "pin.fill")
                        .foregroundColor(.yellow)
                        .font(.caption)
                }
            }

            // Reminder display
            if task.wantsNotification, let reminderDate = task.effectiveReminderTime() {
                HStack {
                    Image(systemName: "bell.fill")
                        .foregroundColor(.blue)
                    Text("Reminder: \(reminderDate, formatter: reminderDateFormatter)")
                        .font(.caption2)
                        .foregroundColor(.blue)
                }
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Helpers

    private func priorityColor(_ priority: TaskPriority) -> Color {
        switch priority {
        case .low: return .green.opacity(0.2)
        case .medium: return .orange.opacity(0.2)
        case .high: return .red.opacity(0.2)
        }
    }

    private var reminderDateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }
}
