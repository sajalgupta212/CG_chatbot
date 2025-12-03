def test_groq_key():
    client = st.session_state.groq_client
    if not client:
        st.warning("Groq client not initialized")
        return
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":"Hello"}],
            max_tokens=10
        )
        st.success("✅ Groq key is valid!")
    except Exception as e:
        st.error(f"❌ Invalid Groq key: {e}")

if st.sidebar.button("✅ Test Groq Key"):
    test_groq_key()
os.getenv("GROQ_API_KEY"))