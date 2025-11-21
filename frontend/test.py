import streamlit as st
import requests

st.set_page_config(page_title="GitHub Viewer Demo", layout="wide")

st.title("🔍 GitHub Page Viewer Demo")
st.write("Three different ways to show GitHub documentation inside Streamlit.")

# --------------------------------------------------------
# Input URL
# --------------------------------------------------------
default_url = "https://github.com/jasp-stats/jaspMixedModels/blob/master/inst/help/MixedModelsBGLMM.md"
github_url = st.text_input("GitHub Markdown URL", default_url)


# --------------------------------------------------------
# Helper: Convert GitHub → raw.githubusercontent URL
# --------------------------------------------------------
def to_raw_url(url: str) -> str:
    return (
        url.replace("https://github.com/", "https://raw.githubusercontent.com/")
           .replace("/blob/", "/")
    )


# --------------------------------------------------------
# Tabs
# --------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📄 Render Markdown (recommended)",
    "🪟 Embed RAW HTML",
    "🖼 Screenshot Preview"
])

# ========================================================
# 1) Render Markdown
# ========================================================
with tab1:
    st.subheader("📄 Rendered Markdown")

    raw_url = to_raw_url(github_url)
    st.code(f"RAW URL: {raw_url}", language="text")

    try:
        res = requests.get(raw_url)
        if res.status_code == 200:
            st.markdown(res.text)
        else:
            st.error(f"Failed to load raw markdown: {res.status_code}")
    except Exception as e:
        st.error(f"Error loading markdown: {e}")


# ========================================================
# 2) Embed RAW HTML (NOT GitHub UI!)
# GitHub blocks iframe for github.com, but raw.githubusercontent.com works
# ========================================================
with tab2:
    st.subheader("🪟 Embedded RAW HTML")

    iframe_url = to_raw_url(github_url)

    st.write("GitHub UI **cannot** be iframed due to security headers.")
    st.write("But the RAW markdown file can be embedded as plain text.")

    st.components.v1.iframe(iframe_url, height=800)


# ========================================================
# 3) Screenshot preview (workaround)
# ========================================================
with tab3:
    st.subheader("🖼 Screenshot preview of GitHub page")
    st.write("GitHub blocks embedding, so screenshot is the only workaround.")

    screenshot_url = f"https://image.thum.io/get/{github_url}"

    st.image(screenshot_url, caption="Screenshot of GitHub page (static)")


st.write("---")
st.success("Demo loaded successfully!")
