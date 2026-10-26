import math
import random
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image  # برای بارگذاری لوگو
import statistics  # برای محاسبات آماری در شبیه‌سازی گروهی

# --------------------------
# تابع کمکی: محدود کردن مقدار بین min و max
# --------------------------
def truncate(value, min_val, max_val):
    return max(min_val, min(max_val, value))

# --------------------------
# تولید داده‌های مصنوعی برای جمعیت ایران با توزیع‌های واقعی‌تر
# --------------------------
def generate_synthetic_data_iran_accurate(n_samples):
    data = []
    random.seed(42)  # بذر تصادفی ثابت برای بازتولید
    for _ in range(n_samples):
        age = truncate(random.gauss(45.0, 12.0), 20.0, 80.0)
        gender = 1 if random.random() < 0.5 else 0
        bmi = truncate(random.gauss(26.0, 4.0), 16.0, 35.0)
        smoker_prob = 0.30 if gender == 1 else 0.05
        smoker = 1 if random.random() < smoker_prob else 0
        diabetes = 1 if random.random() < 0.12 else 0
        job_risk = 1 if random.random() < 0.20 else 0

        # سابقه بیماری‌های خانوادگی و شخصی
        family_history = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5], k=1)[0]
        personal_history = random.choices([0, 1, 2, 3], weights=[80, 10, 7, 3], k=1)[0]
        
        alcohol_drug = 1 if random.random() < 0.15 else 0
        activity_level = random.randint(0, 2)  # 0: کم، 1: متوسط، 2: زیاد
        marital_status = 1 if random.random() < 0.7 else 0
        num_children = random.randint(0, 3) if marital_status == 1 else 0
        housing_status = 1 if random.random() < 0.5 else 0
        # برای آموزش مدل داخلی، همچنان از عدد استفاده می‌کنیم (0 تا 3)
        education_level = random.randint(0, 3)
        
        # ویژگی‌های جدید: درآمد سالانه و بدهی‌های معوق
        annual_income = truncate(random.gauss(50000, 15000), 20000, 150000)
        overdue_debts = truncate(random.gauss(10000, 5000), 0, 50000)

        # محاسبه خطی برای پیش‌بینی ریسک
        linear_sum = (
            -6.0 +
            0.07 * age +
            0.5 * smoker +
            0.5 * diabetes +
            0.6 * job_risk +
            0.5 * family_history +
            0.6 * personal_history +
            0.5 * alcohol_drug +
            (-0.25) * activity_level +
            (-0.1) * marital_status +
            0.05 * num_children +
            (-0.2) * housing_status +
            (-0.1) * education_level +
            (-0.000005) * annual_income +
            (0.00001) * overdue_debts
        )
        mortality_risk = 1.0 / (1.0 + math.exp(-linear_sum))
        mortality_status = 1 if random.random() < mortality_risk else 0

        data.append({
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'smoker': smoker,
            'diabetes': diabetes,
            'job_risk': job_risk,
            'family_history': family_history,
            'personal_history': personal_history,
            'alcohol_drug': alcohol_drug,
            'activity_level': activity_level,
            'marital_status': marital_status,
            'num_children': num_children,
            'housing_status': housing_status,
            'education_level': education_level,
            'annual_income': annual_income,
            'overdue_debts': overdue_debts,
            'mortality_status': mortality_status
        })
    return data

# --------------------------
# کلاس StandardScaler برای نرمال‌سازی ویژگی‌ها
# --------------------------
class StandardScaler:
    def __init__(self):
        self.mean = []
        self.std_dev = []
        self.fitted = False

    def fit(self, X):
        n_features = len(X[0])
        self.mean = [0.0] * n_features
        self.std_dev = [0.0] * n_features
        n_samples = len(X)
        for sample in X:
            for j in range(n_features):
                self.mean[j] += sample[j]
        self.mean = [m / n_samples for m in self.mean]
        for sample in X:
            for j in range(n_features):
                self.std_dev[j] += (sample[j] - self.mean[j]) ** 2
        self.std_dev = [math.sqrt(s / n_samples) for s in self.std_dev]
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            print("خطا: StandardScaler هنوز فیت نشده است!")
            sys.exit(1)
        X_scaled = []
        n_features = len(X[0])
        for sample in X:
            scaled_sample = sample.copy()
            for j in range(n_features):
                if self.std_dev[j] != 0:
                    scaled_sample[j] = (sample[j] - self.mean[j]) / self.std_dev[j]
            X_scaled.append(scaled_sample)
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform_sample(self, sample):
        if not self.fitted:
            print("خطا: StandardScaler هنوز فیت نشده است!")
            sys.exit(1)
        scaled_sample = sample.copy()
        for j in range(len(sample)):
            if self.std_dev[j] != 0:
                scaled_sample[j] = (sample[j] - self.mean[j]) / self.std_dev[j]
        return scaled_sample

# --------------------------
# مدل رگرسیون لجستیک
# --------------------------
class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=3000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def train(self, X, y, progress_callback=None):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        for epoch in range(self.epochs):
            y_pred = [0.0] * n_samples
            for i in range(n_samples):
                z = sum(x * w for x, w in zip(X[i], self.weights)) + self.bias
                y_pred[i] = self.sigmoid(z)
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples
            if progress_callback and (epoch % 100 == 0):
                progress_callback((epoch + 1) / self.epochs)
        if progress_callback:
            progress_callback(1.0)

    def predict(self, X_sample):
        z = sum(x * w for x, w in zip(X_sample, self.weights)) + self.bias
        return self.sigmoid(z)

# --------------------------
# محاسبه مبلغ پرداخت بیمه
# --------------------------
def calculate_payout(mortality_risk, coverage_amount, policy_term_years):
    term_factor = 1.0 + (policy_term_years / 20.0)
    term_factor = min(term_factor, 1.5)
    adjusted_payout = coverage_amount * (1.0 - mortality_risk) * term_factor
    return adjusted_payout

# --------------------------
# ارزیابی مدل روی داده‌های آزمون
# --------------------------
def evaluate_model(model, X_test, y_test):
    correct = 0
    for features, label in zip(X_test, y_test):
        prob = model.predict(features)
        prediction = 1 if prob >= 0.5 else 0
        if prediction == label:
            correct += 1
    return (correct / len(X_test)) * 100.0

# --------------------------
# برنامه گرافیکی با استفاده از customtkinter
# --------------------------
class LifeInsuranceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("دستیار بیمه عمر پیشرفته")
        self.geometry("1100x750")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # تنظیم منوی اصلی
        self.create_menu_bar()

        # پارامترهای مدل (قابل تغییر از طریق تنظیمات پیشرفته)
        self.learning_rate = 0.1
        self.epochs = 3000

        self.content_frame = ctk.CTkFrame(self, corner_radius=15)
        self.content_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=15)
        self.sidebar_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.create_sidebar_buttons()
        self.frames = {}
        self.create_frames() 

        # نوار پیشرفت در داشبورد
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.frames["dashboard"], variable=self.progress_var, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        self.show_frame("dashboard")
        self.train_model_in_background()

        try:
            logo_image = Image.open("logo.png")
            self.logo_image = ctk.CTkImage(logo_image, size=(200, 80))
            self.logo_label = ctk.CTkLabel(self, image=self.logo_image, text="")
            self.logo_label.place(relx=1.0, rely=1.0, anchor="se")
        except Exception as e:
            print("خطا در بارگذاری لوگو:", e)

    # --------------------------
    # ایجاد منوی اصلی (Menu Bar)
    # --------------------------
    def create_menu_bar(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="خروج", command=self.quit)
        menu_bar.add_cascade(label="فایل", menu=file_menu)
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="تنظیمات پیشرفته", command=self.open_advanced_settings)
        menu_bar.add_cascade(label="تنظیمات", menu=settings_menu)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="درباره", command=lambda: self.show_custom_popup("درباره", "این برنامه توسط شرکت کارستان جوان طراحی شده است."))
        menu_bar.add_cascade(label="درباره", menu=help_menu)
        self.config(menu=menu_bar)

    # --------------------------
    # ایجاد دکمه‌های نوار کناری
    # --------------------------
    def create_sidebar_buttons(self):
        self.default_font = ("Segoe UI", 16, "bold")
        self.hover_font = ("Segoe UI", 18, "bold")
        self.dashboard_button = ctk.CTkButton(self.sidebar_frame, text="داشبورد", corner_radius=15,
                                                font=self.default_font, command=self.toggle_dashboard_submenu)
        self.dashboard_button.pack(pady=10, padx=20, fill="x")
        self.dashboard_submenu_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15, fg_color="transparent")
        self.single_button = ctk.CTkButton(self.dashboard_submenu_frame, text="پیش‌بینی تک", corner_radius=15,
                                           font=self.default_font, command=lambda: self.select_submenu("single"))
        self.single_button.pack(pady=5, padx=10, fill="x")
        self.batch_button = ctk.CTkButton(self.dashboard_submenu_frame, text="شبیه‌سازی گروهی", corner_radius=15,
                                          font=self.default_font, command=lambda: self.select_submenu("batch"))
        self.batch_button.pack(pady=5, padx=10, fill="x")
        self.explanation_button = ctk.CTkButton(self.dashboard_submenu_frame, text="توضیح محاسبات", corner_radius=15,
                                                font=self.default_font, command=lambda: self.select_submenu("explanation"))
        self.explanation_button.pack(pady=5, padx=10, fill="x")
        self.dashboard_submenu_visible = False

    def toggle_dashboard_submenu(self):
        if self.dashboard_submenu_visible:
            self.dashboard_submenu_frame.pack_forget()
            self.dashboard_submenu_visible = False
        else:
            self.dashboard_submenu_frame.pack(pady=0, padx=20, fill="x")
            self.dashboard_submenu_visible = True

    def select_submenu(self, frame_key):
        self.show_frame(frame_key)
        self.update_submenu_styles(frame_key)

    def update_submenu_styles(self, selected_key):
        selected_color = "#3B8ED0"
        default_color = "transparent"
        if selected_key == "single":
            self.single_button.configure(fg_color=selected_color)
            self.batch_button.configure(fg_color=default_color)
            self.explanation_button.configure(fg_color=default_color)
        elif selected_key == "batch":
            self.single_button.configure(fg_color=default_color)
            self.batch_button.configure(fg_color=selected_color)
            self.explanation_button.configure(fg_color=default_color)
        elif selected_key == "explanation":
            self.single_button.configure(fg_color=default_color)
            self.batch_button.configure(fg_color=default_color)
            self.explanation_button.configure(fg_color=selected_color)

    def create_frames(self):
        dashboard_frame = ctk.CTkFrame(self.content_frame, corner_radius=15)
        dashboard_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.frames["dashboard"] = dashboard_frame
        self.build_dashboard_frame(dashboard_frame)

        single_frame = ctk.CTkFrame(self.content_frame, corner_radius=15)
        single_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.frames["single"] = single_frame
        self.build_single_frame(single_frame)

        batch_frame = ctk.CTkFrame(self.content_frame, corner_radius=15)
        batch_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.frames["batch"] = batch_frame
        self.build_batch_frame(batch_frame)

        explanation_frame = ctk.CTkFrame(self.content_frame, corner_radius=15)
        explanation_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.frames["explanation"] = explanation_frame
        self.build_explanation_frame(explanation_frame)

    def show_frame(self, frame_key):
        self.frames[frame_key].tkraise()

    def train_model_in_background(self):
        def progress_callback(progress):
            self.progress_var.set(progress)
        def train():
            data = generate_synthetic_data_iran_accurate(1000)
            X = []
            y = []
            for ph in data:
                X.append([
                    ph['age'], ph['gender'], ph['bmi'], ph['smoker'], ph['diabetes'],
                    ph['job_risk'], ph['family_history'], ph['personal_history'],
                    ph['alcohol_drug'], ph['activity_level'], ph['marital_status'],
                    ph['num_children'], ph['housing_status'], ph['education_level'],
                    ph['annual_income'], ph['overdue_debts']
                ])
                y.append(ph['mortality_status'])
            indices = list(range(len(X)))
            random.seed(42)
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            y_shuffled = [y[i] for i in indices]
            split_idx = int(len(X_shuffled) * 0.8)
            self.X_train = X_shuffled[:split_idx]
            self.X_test = X_shuffled[split_idx:]
            self.y_train = y_shuffled[:split_idx]
            self.y_test = y_shuffled[split_idx:]
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.model = LogisticRegression(learning_rate=self.learning_rate, epochs=self.epochs)
            self.model.train(self.X_train_scaled, self.y_train, progress_callback)
            self.test_accuracy = evaluate_model(self.model, self.X_test_scaled, self.y_test)
            self.accuracy_label.configure(text=f"دقت آزمون مدل: {self.test_accuracy:.2f}%")
            self.show_custom_popup("آموزش مدل", "آموزش مدل به پایان رسید.")
        threading.Thread(target=train, daemon=True).start()

    def build_dashboard_frame(self, frame):
        self.accuracy_label = ctk.CTkLabel(frame, text="دقت آزمون مدل: در حال آموزش...", font=("Segoe UI", 16, "bold"), anchor="e")
        self.accuracy_label.pack(pady=20, padx=20, anchor="e")
        refresh_button = ctk.CTkButton(frame, text="به‌روزرسانی دقت", corner_radius=15, font=("Segoe UI", 16, "bold"),
                                       command=self.refresh_accuracy)
        refresh_button.pack(pady=10, padx=20, anchor="e")

    def refresh_accuracy(self):
        self.test_accuracy = evaluate_model(self.model, self.X_test_scaled, self.y_test)
        self.accuracy_label.configure(text=f"دقت آزمون مدل: {self.test_accuracy:.2f}%")
        self.show_custom_popup("داشبورد", "دقت آزمون به‌روزرسانی شد!")

    def show_custom_popup(self, title, message):
        popup = ctk.CTkToplevel(self)
        popup.geometry("600x300")
        popup.title(title)
        popup.lift()
        popup.attributes("-topmost", True)
        popup.after(10, lambda: popup.attributes("-topmost", False))
        label = ctk.CTkLabel(popup, text=message, font=("Segoe UI", 18, "bold"), wraplength=550)
        label.pack(padx=20, pady=20, fill="both", expand=True)
        button = ctk.CTkButton(popup, text="تایید", font=("Segoe UI", 18, "bold"), command=popup.destroy)
        button.pack(pady=10)

    def open_advanced_settings(self):
        settings_popup = ctk.CTkToplevel(self)
        settings_popup.geometry("500x300")
        settings_popup.title("تنظیمات پیشرفته")
        lr_label = ctk.CTkLabel(settings_popup, text="نرخ یادگیری:", font=("Segoe UI", 16, "bold"))
        lr_label.pack(pady=10)
        lr_entry = ctk.CTkEntry(settings_popup, font=("Segoe UI", 16, "bold"))
        lr_entry.insert(0, str(self.learning_rate))
        lr_entry.pack(pady=5)
        epochs_label = ctk.CTkLabel(settings_popup, text="تعداد Epochها:", font=("Segoe UI", 16, "bold"))
        epochs_label.pack(pady=10)
        epochs_entry = ctk.CTkEntry(settings_popup, font=("Segoe UI", 16, "bold"))
        epochs_entry.insert(0, str(self.epochs))
        epochs_entry.pack(pady=5)
        def save_settings():
            try:
                new_lr = float(lr_entry.get())
                new_epochs = int(epochs_entry.get())
            except ValueError:
                self.show_custom_popup("خطا", "ورودی‌های نامعتبر در تنظیمات.")
                return
            self.learning_rate = new_lr
            self.epochs = new_epochs
            self.show_custom_popup("تنظیمات", "تنظیمات جدید ذخیره شد. مدل دوباره آموزش داده خواهد شد.")
            settings_popup.destroy()
            self.train_model_in_background()
        save_button = ctk.CTkButton(settings_popup, text="ذخیره تنظیمات", font=("Segoe UI", 16, "bold"), command=save_settings)
        save_button.pack(pady=20)

    # --------------------------
    # فرم پیش‌بینی تک
    # --------------------------
    def build_single_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        scrollable_container = ctk.CTkScrollableFrame(frame, corner_radius=15)
        scrollable_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        scrollable_container.grid_columnconfigure(0, weight=1)
        scrollable_container.grid_columnconfigure(1, weight=0)
        scrollable_container.grid_columnconfigure(2, weight=1)
        
        fields = [
            "سن",
            "شاخص توده بدنی",
            "جنسیت",
            "سیگاری",
            "ریسک شغلی",
            "سابقه بیماری‌های خانوادگی",
            "سابقه بیماری‌های شخصی",
            "مصرف الکل و مواد مخدر",
            "سطح فعالیت بدنی",
            "وضعیت تأهل",
            "تعداد فرزندان",
            "وضعیت مسکن",
            "سطح تحصیلات",
            "درآمد سالانه",
            "بدهی‌های معوق",
            "مبلغ پوشش",
            "مدت زمان پوشش"
        ]
        
        dropdown_options = {
            "جنسیت": ["زن", "مرد"],
            "سیگاری": ["خیر", "بله"],
            "ریسک شغلی": ["کم", "زیاد"],
            "سابقه بیماری‌های خانوادگی": ["هیچ", "کم", "متوسط", "شدید"],
            "سابقه بیماری‌های شخصی": ["هیچ", "کم", "متوسط", "شدید"],
            "مصرف الکل و مواد مخدر": ["خیر", "بله"],
            "سطح فعالیت بدنی": ["کم", "متوسط", "زیاد"],
            "وضعیت تأهل": ["مجرد", "متأهل"],
            "وضعیت مسکن": ["اجاره", "مالکیت"],
            "سطح تحصیلات": ["بدون مدرک", "دیپلم", "لیسانس", "فوق لیسانس"]
        }
        
        # نگاشت مقادیر متنی به عددی برای ورودی‌های دسته‌بندی شده
        self.dropdown_conversion = {
            "جنسیت": {"زن": 0, "مرد": 1},
            "سیگاری": {"خیر": 0, "بله": 1},
            "ریسک شغلی": {"کم": 0, "زیاد": 1},
            "سابقه بیماری‌های خانوادگی": {"هیچ": 0, "کم": 1, "متوسط": 2, "شدید": 3},
            "سابقه بیماری‌های شخصی": {"هیچ": 0, "کم": 1, "متوسط": 2, "شدید": 3},
            "مصرف الکل و مواد مخدر": {"خیر": 0, "بله": 1},
            "سطح فعالیت بدنی": {"کم": 0, "متوسط": 1, "زیاد": 2},
            "وضعیت تأهل": {"مجرد": 0, "متأهل": 1},
            "وضعیت مسکن": {"اجاره": 0, "مالکیت": 1},
            "سطح تحصیلات": {"بدون مدرک": 0, "دیپلم": 1, "لیسانس": 2, "فوق لیسانس": 3}
        }
        
        widget_width = 200
        widget_height = 40
        self.single_entries = {}
        
        for i, field in enumerate(fields):
            row_frame = ctk.CTkFrame(scrollable_container, corner_radius=15)
            row_frame.grid(row=i, column=0, columnspan=3, pady=5, padx=10, sticky="ew")
            row_frame.grid_columnconfigure(0, weight=0)
            row_frame.grid_columnconfigure(1, weight=1)
            
            if field in dropdown_options:
                widget = ctk.CTkOptionMenu(
                    row_frame,
                    values=dropdown_options[field],
                    font=("Segoe UI", 16, "bold"),
                    width=widget_width,
                    height=widget_height,
                    dropdown_font=("Segoe UI", 16, "bold"),
                    fg_color="#ADD8E6",
                    text_color="black",
                    anchor="w"
                )
            else:
                widget = ctk.CTkEntry(
                    row_frame,
                    width=widget_width,
                    height=widget_height,
                    font=("Segoe UI", 16, "bold"),
                    fg_color="#ADD8E6",
                    text_color="black"
                )
            widget.grid(row=0, column=0, padx=5, sticky="w")
            self.single_entries[field] = widget
            
            lbl = ctk.CTkLabel(
                row_frame,
                text=field,
                font=("Segoe UI", 16, "bold"),
                anchor="e"
            )
            lbl.grid(row=0, column=1, padx=5, sticky="e")
        
        btn_frame = ctk.CTkFrame(scrollable_container)
        btn_frame.grid(row=len(fields), column=0, columnspan=3, pady=15)
        calc_button = ctk.CTkButton(
            btn_frame,
            text="محاسبه",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.calculate_single_prediction
        )
        calc_button.pack(side="left", padx=10)
        clear_button = ctk.CTkButton(
            btn_frame,
            text="پاک کردن",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.clear_single_form
        )
        clear_button.pack(side="left", padx=10)

    def clear_single_form(self):
        for field, widget in self.single_entries.items():
            if hasattr(widget, "delete"):
                widget.delete(0, tk.END)
            elif hasattr(widget, "set") and hasattr(widget, "values"):
                widget.set(widget.values[0])

    def calculate_single_prediction(self):
        try:
            age = float(self.single_entries["سن"].get())
            bmi = float(self.single_entries["شاخص توده بدنی"].get())
            gender = self.dropdown_conversion["جنسیت"][self.single_entries["جنسیت"].get()]
            smoker = self.dropdown_conversion["سیگاری"][self.single_entries["سیگاری"].get()]
            diabetes = 0  # مقدار پیش‌فرض برای دیابت
            job_risk = self.dropdown_conversion["ریسک شغلی"][self.single_entries["ریسک شغلی"].get()]
            family_history = self.dropdown_conversion["سابقه بیماری‌های خانوادگی"][self.single_entries["سابقه بیماری‌های خانوادگی"].get()]
            personal_history = self.dropdown_conversion["سابقه بیماری‌های شخصی"][self.single_entries["سابقه بیماری‌های شخصی"].get()]
            alcohol_drug = self.dropdown_conversion["مصرف الکل و مواد مخدر"][self.single_entries["مصرف الکل و مواد مخدر"].get()]
            activity_level = self.dropdown_conversion["سطح فعالیت بدنی"][self.single_entries["سطح فعالیت بدنی"].get()]
            marital_status = self.dropdown_conversion["وضعیت تأهل"][self.single_entries["وضعیت تأهل"].get()]
            num_children = int(self.single_entries["تعداد فرزندان"].get())
            housing_status = self.dropdown_conversion["وضعیت مسکن"][self.single_entries["وضعیت مسکن"].get()]
            education_level = self.dropdown_conversion["سطح تحصیلات"][self.single_entries["سطح تحصیلات"].get()]
            annual_income = float(self.single_entries["درآمد سالانه"].get())
            overdue_debts = float(self.single_entries["بدهی‌های معوق"].get())
            coverage_amount = float(self.single_entries["مبلغ پوشش"].get())
            policy_term_years = int(self.single_entries["مدت زمان پوشش"].get())
        except ValueError:
            self.show_custom_popup("خطا", "ورودی نامعتبر.")
            return

        features = [
            age, gender, bmi, smoker, diabetes, job_risk,
            family_history, personal_history, alcohol_drug, activity_level,
            marital_status, num_children, housing_status, education_level,
            annual_income, overdue_debts
        ]
        features_scaled = self.scaler.transform_sample(features)
        risk = self.model.predict(features_scaled)
        payout = calculate_payout(risk, coverage_amount, policy_term_years)
        result_text = (f"ریسک مرگ پیش‌بینی‌شده: {risk * 100:.2f}%\n"
                       f"مبلغ پیشنهادی بیمه عمر: ${payout:.2f}")
        self.show_custom_popup("نتیجه پیش‌بینی تک", result_text)

    # --------------------------
    # فرم شبیه‌سازی گروهی
    # --------------------------
    def build_batch_frame(self, frame):
        scrollable_container = ctk.CTkScrollableFrame(frame, corner_radius=15)
        scrollable_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        scrollable_container.grid_columnconfigure(0, weight=1)
        scrollable_container.grid_columnconfigure(1, weight=0)
        scrollable_container.grid_columnconfigure(2, weight=1)
        
        fields = [
            "تعداد پروفایل‌ها",
            "مبلغ پوشش",
            "مدت زمان پوشش"
        ]
        self.batch_entries = {}
        widget_width = 200
        widget_height = 40
        
        for i, field in enumerate(fields):
            row_frame = ctk.CTkFrame(scrollable_container, corner_radius=15)
            row_frame.grid(row=i, column=0, columnspan=3, pady=5, padx=10, sticky="ew")
            row_frame.grid_columnconfigure(0, weight=0)
            row_frame.grid_columnconfigure(1, weight=1)
            row_frame.grid_columnconfigure(2, weight=0)
            
            widget = ctk.CTkEntry(
                row_frame,
                width=widget_width,
                height=widget_height,
                font=("Segoe UI", 16, "bold"),
                fg_color="#ADD8E6",
                text_color="black"
            )
            widget.grid(row=0, column=0, padx=5, sticky="w")
            self.batch_entries[field] = widget
            
            lbl = ctk.CTkLabel(
                row_frame,
                text=field,
                font=("Segoe UI", 16, "bold"),
                anchor="e"
            )
            lbl.grid(row=0, column=1, padx=5, sticky="e")
        
        btn_frame = ctk.CTkFrame(scrollable_container)
        btn_frame.grid(row=len(fields), column=0, columnspan=3, pady=15)
        sim_button = ctk.CTkButton(
            btn_frame,
            text="شبیه‌سازی",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.calculate_batch_simulation
        )
        sim_button.pack(side="left", padx=10)
        clear_button = ctk.CTkButton(
            btn_frame,
            text="پاک کردن",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.clear_batch_form
        )
        clear_button.pack(side="left", padx=10)

    def clear_batch_form(self):
        for entry in self.batch_entries.values():
            entry.delete(0, tk.END)

    def calculate_batch_simulation(self):
        try:
            n = int(self.batch_entries["تعداد پروفایل‌ها"].get())
            coverage_amount = float(self.batch_entries["مبلغ پوشش"].get())
            policy_term_years = int(self.batch_entries["مدت زمان پوشش"].get())
        except ValueError:
            self.show_custom_popup("خطا", "ورودی نامعتبر.")
            return

        risks = []
        payouts = []
        edu_options = ["بدون مدرک", "دیپلم", "لیسانس", "فوق لیسانس"]

        for _ in range(n):
            age = truncate(random.gauss(45.0, 12.0), 20.0, 80.0)
            bmi = truncate(random.gauss(26.0, 4.0), 16.0, 35.0)
            gender = 1 if random.random() < 0.5 else 0
            smoker_prob = 0.30 if gender == 1 else 0.05
            smoker = 1 if random.random() < smoker_prob else 0
            diabetes = 1 if random.random() < 0.12 else 0
            job_risk = 1 if random.random() < 0.20 else 0
            family_history = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5], k=1)[0]
            personal_history = random.choices([0, 1, 2, 3], weights=[80, 10, 7, 3], k=1)[0]
            alcohol_drug = 1 if random.random() < 0.15 else 0
            activity_level = random.randint(0, 2)
            marital_status = 1 if random.random() < 0.7 else 0
            num_children = random.randint(0, 3) if marital_status == 1 else 0
            housing_status = 1 if random.random() < 0.5 else 0
            education_str = random.choice(edu_options)
            education_level = self.dropdown_conversion["سطح تحصیلات"][education_str]
            annual_income = truncate(random.gauss(50000, 15000), 20000, 150000)
            overdue_debts = truncate(random.gauss(10000, 5000), 0, 50000)

            features = [
                age, gender, bmi, smoker, diabetes, job_risk,
                family_history, personal_history, alcohol_drug, activity_level,
                marital_status, num_children, housing_status, education_level,
                annual_income, overdue_debts
            ]
            features_scaled = self.scaler.transform_sample(features)
            risk = self.model.predict(features_scaled)
            payout = calculate_payout(risk, coverage_amount, policy_term_years)
            risks.append(risk)
            payouts.append(payout)

        avg_risk = statistics.mean(risks) * 100
        median_risk = statistics.median(risks) * 100
        stdev_risk = statistics.stdev(risks) * 100 if len(risks) > 1 else 0
        min_risk = min(risks) * 100
        max_risk = max(risks) * 100

        avg_payout = statistics.mean(payouts)
        median_payout = statistics.median(payouts)
        stdev_payout = statistics.stdev(payouts) if len(payouts) > 1 else 0
        min_payout = min(payouts)
        max_payout = max(payouts)

        result_text = (
            "نتیجه شبیه‌سازی گروهی پیشرفته:\n\n"
            f"   ریسک مرگ - میانگین: {avg_risk:.2f}%\n"
            f"   مبلغ پیشنهادی بیمه - میانگین: ${avg_payout:.2f}\n\n"
            f"   ریسک مرگ - میانه: {median_risk:.2f}%\n"
            f"   انحراف معیار ریسک: {stdev_risk:.2f}%\n"
            f"   کمترین ریسک: {min_risk:.2f}%\n"
            f"   بیشترین ریسک: {max_risk:.2f}%\n\n"
            f"   مبلغ بیمه - میانه: ${median_payout:.2f}\n"
            f"   انحراف معیار مبلغ: ${stdev_payout:.2f}\n"
            f"   کمترین مبلغ: ${min_payout:.2f}\n"
            f"   بیشترین مبلغ: ${max_payout:.2f}\n"
        )
        self.show_custom_popup("نتیجه شبیه‌سازی گروهی", result_text)

    # --------------------------
    # فرم توضیح محاسبات
    # --------------------------
    def build_explanation_frame(self, frame):
        textbox = ctk.CTkTextbox(frame, width=600, height=400, font=("Segoe UI", 16, "bold"))
        textbox.pack(padx=20, pady=20, fill="both", expand=True)
        explanation_text = (
            "توضیح محاسبات\n"
            "---------------------------------------------------\n\n"
            "1. محاسبه ریسک مرگ:\n"
            "   فرمول: ریسک = 1 / (1 + exp(-z))\n"
            "   که در آن z = -6.0 + 0.07 * سن + 0.5 * سیگارکش + 0.5 * دیابت + 0.6 * ریسک شغلی\n"
            "       + 0.5 * سابقه بیماری‌های خانوادگی + 0.6 * سابقه بیماری‌های شخصی\n"
            "       + 0.5 * مصرف الکل و مواد مخدر - 0.25 * سطح فعالیت بدنی\n"
            "       - 0.1 * وضعیت تأهل + 0.05 * تعداد فرزندان - 0.2 * وضعیت مسکن\n"
            "       - 0.1 * سطح تحصیلات - 0.000005 * درآمد سالانه + 0.00001 * بدهی‌های معوق\n\n"
            "   مثال:\n"
            "      برای مرد 50 ساله با ویژگی‌های فرضی:\n"
        )
        z_example = (
            -6.0 +
            0.07 * 50 +
            0.5 * 1 +       # سیگاری
            0.5 * 0 +       # دیابت
            0.6 * 1 +       # ریسک شغلی
            0.5 * 1 +       # سابقه بیماری‌های خانوادگی
            0.6 * 0 +       # سابقه بیماری‌های شخصی
            0.5 * 1 +       # مصرف الکل و مواد مخدر
            (-0.25) * 1 +   # سطح فعالیت بدنی
            (-0.1) * 1 +    # وضعیت تأهل
            0.05 * 2 +      # تعداد فرزندان
            (-0.2) * 1 +    # وضعیت مسکن
            (-0.1) * 2 +    # سطح تحصیلات
            (-0.000005) * 50000 +  # درآمد سالانه فرضی
            (0.00001) * 10000      # بدهی‌های معوق فرضی
        )
        risk_example = 1.0 / (1.0 + math.exp(-z_example))
        explanation_text += f"         z = {z_example:.2f}\n"
        explanation_text += f"         ریسک مرگ ≈ {risk_example:.4f}\n\n"
        explanation_text += (
            "2. محاسبه مبلغ بیمه عمر:\n"
            "   فرمول: مبلغ پرداخت = مبلغ پوشش * (1 - ریسک مرگ) * ضریب مدت پوشش\n"
            "   که در آن ضریب مدت پوشش = min(1.0 + (مدت زمان پوشش / 20.0), 1.5)\n\n"
            "   مثال:\n"
            "      برای مبلغ پوشش 100,000 دلار، مدت پوشش 30 سال، ریسک مرگ 0.30:\n"
        )
        term_factor = min(1.0 + (30 / 20.0), 1.5)
        payout_example = 100000 * (1 - 0.30) * term_factor
        explanation_text += f"         ضریب مدت پوشش = {term_factor:.2f}\n"
        explanation_text += f"         مبلغ پرداخت ≈ ${payout_example:.2f}\n\n"
        explanation_text += (
            "3. ویژگی‌های جدید:\n"
            "   - درآمد سالانه: با ضریب -0.000005 در محاسبه ریسک مرگ لحاظ شده است.\n"
            "     افزایش درآمد باعث کاهش ریسک مرگ می‌شود.\n"
            "   - بدهی‌های معوق: با ضریب 0.00001 در محاسبه ریسک مرگ لحاظ شده است.\n"
            "     افزایش بدهی‌های معوق می‌تواند ریسک مرگ را افزایش دهد.\n"
            "   - سطح تحصیلات: به جای اعداد، از مقادیر متنی مانند\n"
            "     'بدون مدرک', 'دیپلم', 'لیسانس' و 'فوق لیسانس' استفاده شده است.\n"
        )
        
        textbox.insert("0.0", explanation_text)
        textbox.configure(state="disabled")

if __name__ == "__main__":
    app = LifeInsuranceApp()
    app.mainloop()
