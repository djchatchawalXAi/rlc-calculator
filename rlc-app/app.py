import streamlit as st
import numpy as np

st.set_page_config(page_title="RLC Circuit Root-Finding", layout="wide")

st.title("⚡ ปฏิบัติการที่ 3.1: ความถี่ที่ให้ขนาดอิมพีแดนซ์ตามกำหนด")
st.write("เครื่องมือคำนวณหารากค่าความถี่เชิงมุม ($\omega$) ของวงจร RLC แบบขนาน ด้วยวิธีเชิงตัวเลขต่างๆ")

# --- Sidebar ตั้งค่าพารามิเตอร์ ---
st.sidebar.header("⚙️ พารามิเตอร์วงจร RLC")
R = st.sidebar.number_input("Resistance R (Ω)", value=225.0)
C = st.sidebar.number_input("Capacitance C (F)", value=0.6e-6, format="%.2e")
L = st.sidebar.number_input("Inductance L (H)", value=0.5)
Z_target = st.sidebar.number_input("Target Impedance Z_target (Ω)", value=75.0)

method = st.sidebar.selectbox("เลือกวิธีคำนวณ (Method)", ["Bisection", "Newton-Raphson", "Secant", "เปรียบเทียบทั้งหมด"])

st.sidebar.markdown("---")
st.sidebar.header("🎚️ กำหนดช่วงเริ่มต้น (Intervals)")
region = st.sidebar.radio("เลือกฝั่งความถี่ของเรโซแนนซ์", ["ฝั่งซ้าย (omega < omega_0)", "ฝั่งขวา (omega > omega_0)"])

if "ซ้าย" in region:
    default_a, default_b, default_x0, default_x1 = 1200.0, 1600.0, 1400.0, 1200.0
else:
    default_a, default_b, default_x0, default_x1 = 2100.0, 2500.0, 2200.0, 2100.0

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    param1 = st.number_input("a / x0", value=default_a)
with col_s2:
    param2 = st.number_input("b / x1", value=default_b)

max_iter = st.sidebar.slider("Max Iterations", 10, 200, 100)
x_tol = st.sidebar.slider("Tolerance (x_tol)", 1e-8, 1e-4, 1e-6, format="%.1e")

# --- ฟังก์ชันคำนวณ RLC ---
def B(omega):
    return omega * C - 1.0 / (omega * L)

def Z_func(omega):
    if omega <= 0:
        return np.inf
    return 1.0 / np.sqrt((1.0 / R)**2 + B(omega)**2)

def f(omega):
    if omega <= 0:
        return np.nan
    return Z_func(omega) - Z_target

def df(omega):
    if omega <= 0:
        return np.nan
    b_val = B(omega)
    z_val = Z_func(omega)
    b_prime = C + 1.0 / (omega**2 * L)
    return - (z_val**3) * b_val * b_prime

# --- อัลกอริทึม Bisection ---
def run_bisection(a, b):
    history = []
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb) or a <= 0 or b <= 0:
        return None, "out_of_domain", 0, history
    if fa * fb > 0:
        return None, "invalid_bracket", 0, history
    for k in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        history.append({"Iteration": k, "a": a, "b": b, "Root (m)": m, "f(m)": fm})
        if 0.5 * (b - a) < x_tol or abs(fm) <= x_tol:
            return m, "converged", k, history
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b), "max_iter", max_iter, history

# --- อัลกอริทึม Newton ---
def run_newton(x0):
    history = []
    xk = x0
    if xk <= 0:
        return None, "out_of_domain", 0, history
    for k in range(1, max_iter + 1):
        fxk = f(xk)
        dfxk = df(xk)
        history.append({"Iteration": k, "x_k": xk, "f(x_k)": fxk, "|delta_x|": 0.0})
        if abs(fxk) <= x_tol:
            return xk, "converged", k, history
        if dfxk == 0 or np.isnan(dfxk):
            return xk, "zero_denominator", k, history
        delta_x = fxk / dfxk
        x_next = xk - delta_x
        if x_next <= 0:
            return x_next, "out_of_domain", k, history
        history[-1]["|delta_x|"] = abs(delta_x)
        if abs(delta_x) < x_tol:
            return x_next, "converged", k, history
        xk = x_next
    return xk, "max_iter", max_iter, history

# --- อัลกอริทึม Secant ---
def run_secant(x0, x1):
    history = []
    if x0 <= 0 or x1 <= 0:
        return None, "out_of_domain", 0, history
    f0, f1 = f(x0), f(x1)
    for k in range(1, max_iter + 1):
        history.append({"Iteration": k, "x_k": x1, "f(x_k)": f1, "|x_k - x_{k-1}|": abs(x1 - x0)})
        if abs(f1) <= x_tol:
            return x1, "converged", k, history
        denominator = (f1 - f0)
        if denominator == 0 or np.isnan(denominator):
            return x1, "zero_denominator", k, history
        x_next = x1 - f1 * (x1 - x0) / denominator
        if x_next <= 0:
            return x_next, "out_of_domain", k, history
        if abs(x_next - x1) < x_tol:
            return x_next, "converged", k, history
        x0, f0 = x1, f1
        x1, f1 = x_next, f(x_next)
    return x1, "max_iter", max_iter, history

# --- ปุ่มสั่งคำนวณ ---
if st.button("🚀 รันการคำนวณ", type="primary"):
    st.markdown("---")
    if method == "Bisection":
        root, status, iters, hist = run_bisection(param1, param2)
        st.subheader(f"ผลลัพธ์วิธี Bisection (สถานะ: `{status}`)")
        st.metric(label="ค่ารากที่ได้ ($\omega$)", value=f"{root:.6f} rad/s" if root else "N/A")
        st.write(f"จำนวนรอบ (Iterations): {iters}")
        if hist:
            st.dataframe(hist)
            
    elif method == "Newton-Raphson":
        root, status, iters, hist = run_newton(param1)
        st.subheader(f"ผลลัพธ์วิธี Newton-Raphson (สถานะ: `{status}`)")
        st.metric(label="ค่ารากที่ได้ ($\omega$)", value=f"{root:.6f} rad/s" if root else "N/A")
        st.write(f"จำนวนรอบ (Iterations): {iters}")
        if hist:
            st.dataframe(hist)
            
    elif method == "Secant":
        root, status, iters, hist = run_secant(param1, param2)
        st.subheader(f"ผลลัพธ์วิธี Secant (สถานะ: `{status}`)")
        st.metric(label="ค่ารากที่ได้ ($\omega$)", value=f"{root:.6f} rad/s" if root else "N/A")
        st.write(f"จำนวนรอบ (Iterations): {iters}")
        if hist:
            st.dataframe(hist)
            
    else:
        st.subheader("📊 เปรียบเทียบทุกวิธี")
        rb, sb, ib, _ = run_bisection(param1, param2)
        rn, sn, in_, _ = run_newton(param1)
        rs, ss, is_, _ = run_secant(param1, param2)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Bisection")
            st.metric("Root", f"{rb:.4f}" if rb else "N/A")
            st.write(f"Status: `{sb}` ({ib} รอบ)")
        with col2:
            st.markdown("### Newton-Raphson")
            st.metric("Root", f"{rn:.4f}" if rn else "N/A")
            st.write(f"Status: `{sn}` ({in_} รอบ)")
        with col3:
            st.markdown("### Secant")
            st.metric("Root", f"{rs:.4f}" if rs else "N/A")
            st.write(f"Status: `{ss}` ({is_} รอบ)")