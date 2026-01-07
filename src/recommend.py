# src/recommend.py
"""
Curated playlist recommendations with brand-consistent buttons.
Clean, minimal, professional Streamlit UI.
"""

import streamlit as st

GENRE_PLAYLISTS = {
    "rock": {
        "spotify": "https://open.spotify.com/album/7sm8hEjRgCfzWoRPPQWeam",
        "youtube": "https://youtube.com/playlist?list=PLn4GvABOzCQursVQ7qMU9CkNaKz4RgrVM"
    },
    "blues": {
        "spotify": "https://open.spotify.com/playlist/0A1IHcqjyImN9uoHRsVtBn",
        "youtube": "https://youtube.com/playlist?list=PLjzeyhEA84sQKuXp-rpM1dFuL2aQM_a3S"
    },
    "classical": {
        "spotify": "https://open.spotify.com/playlist/27Zm1P410dPfedsdoO9fqm",
        "youtube": "https://youtube.com/playlist?list=PLcGkkXtask_fpbK9YXSzlJC4f0nGms1mI"
    },
    "country": {
        "spotify": "https://open.spotify.com/playlist/37i9dQZF1EQmPV0vrce2QZ",
        "youtube": "https://youtube.com/playlist?list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S"
    },
    "disco": {
        "spotify": "https://open.spotify.com/playlist/2H2VOvgUb0ffkYjOKRiWF7",
        "youtube": "https://youtube.com/playlist?list=PLEXox2R2RxZKUmrWKNF61K-kZSov14Snr"
    },
    "hiphop": {
        "spotify": "https://open.spotify.com/playlist/37i9dQZF1EQnqst5TRi17F",
        "youtube": "https://youtube.com/playlist?list=PLOhV0FrFphUdkuWPE2bzJEsGxXMRKVkoM"
    },
    "jazz": {
        "spotify": "https://open.spotify.com/album/4nY56CUHJuJMAWv3TumBC7",
        "youtube": "https://youtube.com/playlist?list=PL8F6B0753B2CCA128"
    },
    "metal": {
        "spotify": "https://open.spotify.com/playlist/1GXRoQWlxTNQiMNkOe7RqA",
        "youtube": "https://youtube.com/playlist?list=PLhQCJTkrHOwSX8LUnIMgaTq3chP1tiTut"
    },
    "pop": {
        "spotify": "https://open.spotify.com/playlist/1WH6WVBwPBz35ZbWsgCpgr",
        "youtube": "https://youtube.com/playlist?list=PLa2a9FJY91_0x1s4eq6mf9b91m3Pahv0G"
    },
    "reggae": {
        "spotify": "https://open.spotify.com/playlist/37i9dQZF1E8Ec8HHX0pz0d",
        "youtube": "https://youtube.com/playlist?list=PLw7aLrPJ8Hl24kczxvRoPECfTvAMsOwv1"
    }
}

def show_recommendations(genre: str):
    genre = genre.lower()
    if genre not in GENRE_PLAYLISTS:
        st.warning("No playlist available for this genre.")
        return

    p = GENRE_PLAYLISTS[genre]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <a href="{p['spotify']}" target="_blank" style="text-decoration:none;">
                <div style="
                    background:#1DB954;
                    color:black;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    font-weight:600;
                ">
                    Open on Spotify
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <a href="{p['youtube']}" target="_blank" style="text-decoration:none;">
                <div style="
                    background:#FF0000;
                    color:white;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    font-weight:600;
                ">
                    Open on YouTube
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )
