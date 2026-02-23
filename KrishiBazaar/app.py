from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from googletrans import Translator
from werkzeug.utils import secure_filename
import pandas as pd
import os
import random
import json

# IMPORT PROPHET FOR MACHINE LEARNING
from prophet import Prophet

# IMPORT SMTP FOR SMS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# IMPORT MONGOENGINE
from mongoengine import connect, Document, StringField, FloatField, IntField, ReferenceField, DateTimeField, Q, BooleanField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'

# --- CONNECT TO MONGODB ---
connect('farm_db', host='localhost:27017')

# --- IMAGE UPLOAD CONFIG ---
UPLOAD_FOLDER = 'static/product_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- SMTP (EMAIL-TO-SMS) CONFIGURATION ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"  
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx" 

CARRIERS = {
    "verizon": "@vtext.com",
    "att": "@txt.att.net",
    "tmobile": "@tmomail.net",
    "sprint": "@messaging.sprintpcs.com"
}

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
translator = Translator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_sms_via_email(phone_number, carrier, message):
    try:
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        if len(clean_phone) > 10: clean_phone = clean_phone[-10:] 
        
        gateway = CARRIERS.get(carrier.lower())
        if not gateway: return False

        to_address = f"{clean_phone}{gateway}"
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_address
        msg['Subject'] = "KrishiBazaar OTP"
        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_address, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"SMTP Error: {e}")
        return False

# --- MONGODB MODELS ---
class User(Document, UserMixin):
    username = StringField(required=True, unique=True)
    phone = StringField(required=True, unique=True)
    password = StringField(required=True)
    role = StringField(required=True)

class Product(Document):
    name = StringField(required=True)
    price = FloatField(required=True)
    quantity = IntField(required=True)
    category = StringField()
    location = StringField(default='Not specified')
    image = StringField(default='default.jpg')
    farmer = ReferenceField(User)
    
    accepts_cod = BooleanField(default=True)
    accepts_upi = BooleanField(default=False)
    upi_qr = StringField(default='')

class Order(Document):
    product = ReferenceField(Product)
    consumer = ReferenceField(User)
    farmer = ReferenceField(User)
    quantity = IntField(default=1)
    status = StringField(default='Pending')
    date = DateTimeField(default=datetime.utcnow)
    
    payment_method = StringField(default='Cash on Delivery')

class ActivityLog(Document):
    user = ReferenceField(User)
    action = StringField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.objects(id=user_id).first()

# --- PROPHET MACHINE LEARNING AI ---
class DemandForecaster:
    def __init__(self, data_file='historical_data.csv'):
        self.data_file = data_file

    def predict_all(self, target_date_str):
        if not os.path.exists(self.data_file):
            return {"error": "Historical data file missing."}
        df = pd.read_csv(self.data_file)
        if df.empty or 'product_name' not in df.columns:
            return {"error": "No historical data available."}

        try:
            target_date = pd.to_datetime(target_date_str)
            future = pd.DataFrame({'ds': [target_date]})
            results = []
            unique_products = df['product_name'].unique()
            
            for prod in unique_products:
                prod_df = df[df['product_name'] == prod].copy()
                prod_df['ds'] = pd.to_datetime(prod_df['date'])
                
                qty_df = prod_df[['ds', 'quantity_sold']].rename(columns={'quantity_sold': 'y'})
                m_qty = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m_qty.fit(qty_df)
                pred_qty = int(max(0, m_qty.predict(future)['yhat'].iloc[0])) 
                
                price_df = prod_df[['ds', 'price_per_kg']].rename(columns={'price_per_kg': 'y'})
                m_price = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m_price.fit(price_df)
                pred_price = round(m_price.predict(future)['yhat'].iloc[0], 2)

                results.append({"product": str(prod).capitalize(), "predicted_qty": pred_qty, "predicted_price": pred_price})
            return {"date": target_date.strftime('%B %d, %Y'), "predictions": results}
        except Exception as e:
            return {"error": "Failed to generate forecast."}

    def predict_single(self, product_name, target_date_str):
        if not os.path.exists(self.data_file):
            return {"error": "No data"}
        df = pd.read_csv(self.data_file)
        prod_df = df[df['product_name'].str.lower() == product_name.lower()].copy()
        
        if prod_df.empty:
            return {"error": "No data"}

        try:
            target_date = pd.to_datetime(target_date_str)
            future = pd.DataFrame({'ds': [target_date]})
            prod_df['ds'] = pd.to_datetime(prod_df['date'])
            
            qty_df = prod_df[['ds', 'quantity_sold']].rename(columns={'quantity_sold': 'y'})
            m_qty = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m_qty.fit(qty_df)
            pred_qty = int(max(0, m_qty.predict(future)['yhat'].iloc[0]))
            
            price_df = prod_df[['ds', 'price_per_kg']].rename(columns={'price_per_kg': 'y'})
            m_price = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m_price.fit(price_df)
            pred_price = round(m_price.predict(future)['yhat'].iloc[0], 2)
            
            return {"predicted_qty": pred_qty, "predicted_price": pred_price}
        except Exception as e:
            return {"error": str(e)}

forecaster = DemandForecaster()

# --- TRANSLATION ---
@app.context_processor
def inject_translator():
    def translate_text(text):
        try:
            dest_lang = session.get('lang', 'en')
            if dest_lang == 'en': return text
            return translator.translate(text, dest=dest_lang).text
        except: return text
    return dict(translate=translate_text)

@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    session['lang'] = lang_code
    return redirect(request.referrer or url_for('index'))

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_result = None
    if request.method == 'POST':
        target_date = request.form.get('target_date')
        if target_date:
            forecast_result = forecaster.predict_all(target_date)
            
    search_query = request.args.get('q', '')
    if search_query:
        products = list(Product.objects(
            (Q(name__icontains=search_query) | Q(category__icontains=search_query) | Q(location__icontains=search_query)) & Q(quantity__gt=0)
        ))
    else:
        products = list(Product.objects(quantity__gt=0).limit(6))
        
    return render_template('index.html', products=products, forecast_result=forecast_result, search_query=search_query)

@app.route('/api/forecast', methods=['GET'])
@login_required
def api_forecast():
    product = request.args.get('product')
    if not product:
        return jsonify({"error": "No product provided"})
    
    today = datetime.now().strftime('%Y-%m-%d')
    result = forecaster.predict_single(product, today)
    return jsonify(result)

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/customer_service')
def customer_service(): return render_template('customer_service.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        phone = request.form.get('phone')
        password = request.form.get('password')
        role = request.form.get('role')
        if User.objects(Q(username=username) | Q(phone=phone)).first():
            flash('Username or Phone number already exists.')
            return redirect(url_for('register'))
        new_user = User(username=username, phone=phone, password=password, role=role)
        new_user.save() 
        ActivityLog(user=new_user, action=f'Registered as {role}').save()
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.objects(username=username).first()
        if user and user.password == password:
            login_user(user)
            ActivityLog(user=user, action='Logged in via password').save()
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        phone = request.form.get('phone')
        carrier = request.form.get('carrier')
        user = User.objects(phone=phone).first()
        if user:
            otp = random.randint(1000, 9999)
            session['otp'] = otp
            session['reset_user_id'] = str(user.id)
            if "your_email@gmail.com" in SENDER_EMAIL:
                flash(f"SMTP not configured. Test Mode OTP: {otp}")
                return redirect(url_for('verify_otp'))
            success = send_sms_via_email(phone, carrier, f"Your KrishiBazaar OTP is: {otp}")
            if success:
                flash(f'OTP sent to {phone} via email gateway!')
            else:
                flash(f"Failed to send SMS. Test Mode OTP: {otp}")
            return redirect(url_for('verify_otp'))
        else:
            flash('Phone number not found.')
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        saved_otp = session.get('otp')
        if saved_otp and int(entered_otp) == saved_otp:
            user_id = session.get('reset_user_id')
            user = User.objects(id=user_id).first()
            login_user(user)
            session.pop('otp', None)
            ActivityLog(user=user, action='Logged in via OTP').save()
            flash('OTP Verified! Logged in successfully.')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid OTP. Please try again.')
    return render_template('verify_otp.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'farmer':
        my_products = list(Product.objects(farmer=current_user))
        incoming_orders = list(Order.objects(farmer=current_user))
        total_sales = Order.objects(farmer=current_user, status='Accepted').count()
        return render_template('farmer_dashboard.html', products=my_products, orders=incoming_orders, sales=total_sales)
    else:
        search_query = request.args.get('q', '')
        if search_query:
            products = list(Product.objects(
                Q(name__icontains=search_query) | Q(category__icontains=search_query) | Q(location__icontains=search_query)
            ))
        else:
            products = list(Product.objects())
        my_orders = list(Order.objects(consumer=current_user))
        return render_template('consumer_dashboard.html', products=products, orders=my_orders, search_query=search_query)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Unauthorized Access!')
        return redirect(url_for('dashboard'))
    users = list(User.objects())
    products = list(Product.objects())
    orders = list(Order.objects().order_by('-date'))
    logs = list(ActivityLog.objects().order_by('-timestamp').limit(100))
    total_sales_kg = sum([o.quantity for o in orders if o.status == 'Accepted'])
    total_revenue = sum([(o.quantity * o.product.price) for o in orders if o.status == 'Accepted'])
    total_inventory = sum([p.quantity for p in products])
    return render_template('admin_dashboard.html', users=users, products=products, orders=orders, logs=logs, total_sales_kg=total_sales_kg, total_revenue=total_revenue, total_inventory=total_inventory)

@app.route('/add_product', methods=['POST'])
@login_required
def add_product():
    if current_user.role != 'farmer': return redirect(url_for('index'))
    name = request.form.get('name')
    price = float(request.form.get('price'))
    qty = int(request.form.get('quantity'))
    category = request.form.get('category')
    location = request.form.get('location')
    
    accepts_cod = 'cod' in request.form
    accepts_upi = 'upi' in request.form
    if not accepts_cod and not accepts_upi:
        accepts_cod = True 
        
    image_file = request.files.get('image')
    filename = 'default.jpg'
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
    qr_filename = ''
    qr_file = request.files.get('upi_qr')
    if accepts_upi and qr_file and allowed_file(qr_file.filename):
        qr_filename = secure_filename(qr_file.filename)
        qr_filename = f"qr_{datetime.now().strftime('%Y%m%d%H%M%S')}_{qr_filename}"
        qr_file.save(os.path.join(app.config['UPLOAD_FOLDER'], qr_filename))
    
    new_prod = Product(name=name, price=price, quantity=qty, category=category, location=location, image=filename, farmer=current_user, accepts_cod=accepts_cod, accepts_upi=accepts_upi, upi_qr=qr_filename)
    new_prod.save()
    ActivityLog(user=current_user, action=f'Added new product: {name}').save()
    flash(f'Product Added!')
    return redirect(url_for('dashboard'))

# --- NEW: EDIT PRODUCT ROUTE ---
@app.route('/update_product/<string:product_id>', methods=['POST'])
@login_required
def update_product(product_id):
    if current_user.role != 'farmer': 
        return redirect(url_for('index'))
    
    product = Product.objects(id=product_id, farmer=current_user).first()
    
    if product:
        try:
            new_price = float(request.form.get('price'))
            new_qty = int(request.form.get('quantity'))
            
            product.price = new_price
            product.quantity = new_qty
            product.save()
            
            ActivityLog(user=current_user, action=f'Updated price/qty for {product.name}').save()
            flash(f'{product.name} updated successfully!')
        except Exception as e:
            flash('Error updating product. Please check your inputs.')
            
    return redirect(url_for('dashboard'))

@app.route('/buy/<string:product_id>', methods=['POST'])
@login_required
def buy_product(product_id):
    product = Product.objects(id=product_id).first()
    order_qty = int(request.form.get('order_quantity', 1))
    if product and product.quantity >= order_qty and order_qty > 0:
        new_order = Order(product=product, consumer=current_user, farmer=product.farmer, quantity=order_qty)
        new_order.save()
        ActivityLog(user=current_user, action=f'Placed order for {order_qty}kg of {product.name}').save()
        flash(f'Order placed for {order_qty} kg of {product.name}!')
    else:
        flash('Invalid quantity or out of stock!')
    return redirect(url_for('dashboard'))

@app.route('/manage_order/<string:order_id>/<action>')
@login_required
def manage_order(order_id, action):
    order = Order.objects(id=order_id).first()
    if not order or order.farmer.id != current_user.id: return "Unauthorized"
    if action == 'accept':
        if order.product.quantity >= order.quantity:
            order.status = 'Accepted'
            order.product.quantity -= order.quantity
            order.product.save()
            ActivityLog(user=current_user, action=f'Accepted order #{order.id}').save()
            flash('Order Accepted!')
        else:
            flash("Not enough stock left to accept this order!")
    elif action == 'reject': 
        order.status = 'Rejected'
        ActivityLog(user=current_user, action=f'Rejected order #{order.id}').save()
        flash('Order Rejected.')
    order.save()
    return redirect(url_for('dashboard'))

@app.route('/pay_via_upi/<string:order_id>', methods=['POST'])
@login_required
def pay_via_upi(order_id):
    order = Order.objects(id=order_id).first()
    if order and order.consumer.id == current_user.id:
        order.payment_method = 'UPI'
        order.save()
        ActivityLog(user=current_user, action=f'Paid via UPI for order #{order.id}').save()
        flash('UPI Payment Confirmed Successfully!')
    return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    ActivityLog(user=current_user, action='Logged out').save()
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        admin_user = User.objects(username='Subhajit Rudra').first()
        if not admin_user:
            new_admin = User(username='Subhajit Rudra', phone='Admin', password='Subhajit2005', role='admin')
            new_admin.save()
            print("Master Admin Account 'Subhajit Rudra' successfully generated.")
    except Exception as e:
        print(f"MongoDB Error: {e}")
    app.run(debug=True)