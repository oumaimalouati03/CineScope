from pathlib import Path
import sys
import pandas as pd
import streamlit as st

st.set_page_config(page_title='CineScope — Découverte cinéma', page_icon='🎬', layout='wide', initial_sidebar_state='expanded')
st.markdown(
    """
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from recommender import MovieRecommender
import database as db

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#f4f3ef; --muted:#a9a9b4; --accent:#d7f36b; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #252b29 0%, #141719 44%, #101214 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
[data-testid="stSidebar"] { background:#171a1b; border-right:1px solid #303536; }
[data-testid="stMetric"] { background:#1c2020; padding:16px; border:1px solid #343a35; border-radius:14px; }
.stButton>button { border-radius:999px; border:1px solid #424847; background:#222726; color:#f4f3ef; }
.stButton>button:hover { border-color:#d7f36b; color:#d7f36b; }
[data-testid="stForm"] { border:1px solid #343a35; border-radius:14px; }
section[data-testid="stMain"] .block-container { padding-top:2rem; max-width:1400px; }
.small-muted { color:#a9a9b4; font-size:.9rem; }
.hero { padding:28px 30px; border:1px solid #3a413a; border-radius:22px; background:linear-gradient(120deg,rgba(215,243,107,.13),rgba(35,43,39,.2) 65%),#1a1f1d; margin-bottom:22px; }
.hero-kicker { color:#d7f36b; text-transform:uppercase; letter-spacing:.16em; font-size:.75rem; font-weight:700; }
.hero h1 { font-size:2.8rem; margin:.35rem 0; }
.movie-card { padding:15px; border:1px solid #353b39; background:#1a1e1e; border-radius:15px; min-height:130px; }
.movie-card h4 { margin:.4rem 0; font-size:1rem; }
.movie-meta { color:#a9a9b4; font-size:.8rem; }
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    movies = pd.read_csv(ROOT / 'data' / 'movies_clean.csv')
    ratings = pd.read_csv(ROOT / 'data' / 'ratings_clean.csv')
    tags = pd.read_csv(ROOT / 'data' / 'tags_clean.csv')
    return MovieRecommender(movies, ratings, tags)
engine = load_engine()
con = db.connect()

with st.sidebar:
    st.markdown('## 🎬 CineScope')
   
    username = st.text_input('Ton profil', value=st.session_state.get('username','ciné-oumaima'), max_chars=30).strip()
    username = username or 'ciné-oumaima'
    st.session_state['username'] = username
    db.ensure_profile(con, username)
    page = st.radio('Navigation', ['Accueil','Découvrir les films','Pour moi','Ma collection','Historique','À propos'], label_visibility='collapsed')
    st.divider()
    

username = st.session_state['username']
local_ratings = db.get_ratings(con, username)
local_favorites = db.get_favorites(con, username)
local_history = db.get_history(con, username)
liked_ids = [int(r['movie_id']) for r in local_ratings if float(r['rating']) >= 4]
liked_ids += [int(r['movie_id']) for r in local_favorites]


def movie_label(mid):
    row = engine.movies.loc[engine.movies.movieId == int(mid)]
    return row.iloc[0].title if not row.empty else f'Film #{mid}'

def render_movie_grid(df, prefix='movie'):
    if df.empty:
        st.info('Aucun film trouvé pour ces critères.')
        return
    cols = st.columns(4)
    for i, (_, film) in enumerate(df.iterrows()):
        with cols[i % 4]:
            genres = str(film.get('genres','')).replace('|',' · ')
            year = film.get('year', '')
            if pd.isna(year): year = ''
            rating = film.get('avg_rating', 0)
            score = film.get('score', None)
            st.markdown(f'''<div class="movie-card"><div style="font-size:2rem">🎞️</div><h4>{film['title']}</h4><div class="movie-meta">{year} {'· ' + genres if genres else ''}</div><div style="margin-top:8px">⭐ {float(rating):.1f}/5 {' · ✦ ' + str(round(float(score)*100))+'% match' if score is not None else ''}</div></div>''', unsafe_allow_html=True)
            with st.expander('Détails & actions', expanded=False):
                st.write(f"**Genres :** {genres or 'Non renseignés'}")
                st.write(f"**Nombre de notes :** {int(film.get('rating_count',0))}")
                if 'reason' in film: st.caption(film['reason'])
                mid = int(film['movieId'])
                if st.button('☆ Ajouter / retirer des favoris', key=f'{prefix}_fav_{mid}', use_container_width=True):
                    added = db.toggle_favorite(con, username, mid)
                    st.toast('Ajouté aux favoris' if added else 'Retiré des favoris')
                    st.rerun()
                with st.form(f'{prefix}_rate_{mid}'):
                    rating_val = st.slider('Ta note', 0.5, 5.0, 4.0, 0.5, key=f'{prefix}_slider_{mid}')
                    if st.form_submit_button('Enregistrer ma note', use_container_width=True):
                        db.save_rating(con, username, mid, rating_val)
                        st.toast('Note enregistrée dans ton profil')
                        st.rerun()
                if st.button('Marquer comme consulté', key=f'{prefix}_view_{mid}', use_container_width=True):
                    db.log_view(con, username, mid)
                    st.toast('Ajouté à ton historique')
                    st.rerun()

if page == 'Accueil':
    st.markdown('''<div class="hero"><div class="hero-kicker">Ton prochain film préféré commence ici</div><h1>Le cinéma, à ton image.</h1><p class="small-muted">Explore des milliers de films. Construis ta collection. Découvre ton prochain coup de cœur grâce à des recommandations personnalisées.</p></div>''', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Films au catalogue', f'{len(engine.movies):,}'.replace(',',' '))
    c2.metric('Notes utilisées', f'{len(engine.ratings):,}'.replace(',',' '))
    c3.metric('Tes films notés', len(local_ratings))
    c4.metric('Tes favoris', len(local_favorites))
    st.subheader('✦ À découvrir pour toi')
    recs = engine.recommend(liked_movie_ids=liked_ids, limit=8)
    render_movie_grid(recs, 'home')
    st.subheader('Populaires auprès de la communauté')
    render_movie_grid(engine.search('', limit=8), 'popular')

elif page == 'Découvrir les films':
    st.title('Découvrir les films')
    st.write('Recherche un titre ou explore le catalogue par genre.')
    genres = sorted({g for item in engine.movies.genres.fillna('') for g in item.split('|') if g and g != '(no genres listed)'})
    c1,c2 = st.columns([3,1])
    query = c1.text_input('Titre du film', placeholder='Ex. Toy Story, Matrix, Titanic…')
    genre = c2.selectbox('Genre', ['Tous'] + genres)
    results = engine.search(query, genre, limit=40)
    st.caption(f'{len(results)} résultat(s) affiché(s) au maximum')
    render_movie_grid(results, 'discover')

elif page == 'Pour moi':
    st.title('Des films faits pour toi')

    mode = st.radio(
        'Type de recommandations',
        ['Hybrides personnalisées', 'Basées sur mes favoris', 'Populaires'],
        horizontal=True
    )

    if mode == 'Populaires':
        recs = engine.recommend(limit=24)
    elif mode == 'Basées sur mes favoris':
        recs = engine.recommend(liked_movie_ids=liked_ids, limit=24)
    else:
        # Utilise les notes locales comme profil temporaire, sans modifier les données source.
        if local_ratings:
            temp = pd.DataFrame([
                {
                    'userId': -1,
                    'movieId': int(r['movie_id']),
                    'rating': float(r['rating'])
                }
                for r in local_ratings
            ])
            augmented = pd.concat([engine.ratings, temp], ignore_index=True)
            personalized = MovieRecommender(engine.movies, augmented, engine.tags)
            recs = personalized.recommend(
                user_id=-1,
                liked_movie_ids=liked_ids,
                limit=24
            )
        else:
            # Repli pour un utilisateur sans notes enregistrées.
            recs = engine.recommend(liked_movie_ids=liked_ids, limit=24)

    render_movie_grid(recs, 'for_you')

elif page == 'Ma collection':
    st.title('Ma collection')
    tab1,tab2 = st.tabs(['♥ Favoris','★ Mes notes'])
    with tab1:
        if not local_favorites: st.info('Ta liste est vide. Ajoute des films en favoris depuis leurs détails.')
        else:
            df = pd.DataFrame([engine.movie_details(int(r['movie_id'])) for r in local_favorites])
            df['movieId'] = df.movieId.astype(int)
            render_movie_grid(df, 'collection_fav')
    with tab2:
        if not local_ratings: st.info('Tu n’as pas encore noté de films.')
        else:
            df = pd.DataFrame([engine.movie_details(int(r['movie_id'])) for r in local_ratings])
            df['movieId'] = df.movieId.astype(int)
            rating_map = {int(r['movie_id']):float(r['rating']) for r in local_ratings}
            df['avg_rating'] = df.movieId.map(rating_map)
            df = df.rename(columns={'avg_rating':'avg_rating'})
            render_movie_grid(df, 'collection_rating')

elif page == 'Historique':
    st.title('Mon historique')
    st.write('Les consultations, changements de favoris et notes enregistrées dans cette application apparaissent ici.')
    if not local_history: st.info('Ton historique est encore vide. Consulte un film ou interagis avec sa fiche pour commencer.')
    else:
        hist_df = pd.DataFrame([{'Film':movie_label(r['movie_id']),'Action':r['action'],'Date':r['created_at']} for r in local_history])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        


else:
    st.title("🎬 À propos de CineScope")

    st.markdown(
        """
        <div style="text-align: center; padding: 20px 10px;">
            <h2>Découvrez le cinéma autrement.</h2>
            <p style="font-size: 17px;">
                Bienvenue sur <strong>CineScope</strong>, votre espace dédié à la
                découverte, à la passion et au partage autour du cinéma.
            </p>
            <p>
                Notre objectif est de vous aider à découvrir des films adaptés
                à vos goûts grâce à une expérience personnalisée et intuitive.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(" Ce que nous proposons")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("###  Découvrir des films")
            st.write(
                "Explorez une collection de films de genres et d'époques "
                "différents pour découvrir de nouvelles histoires."
            )

        with st.container(border=True):
            st.markdown("###  Noter et ajouter aux favoris")
            st.write(
                "Notez les films que vous avez regardés et conservez "
                "vos préférés dans votre collection personnelle."
            )

    with col2:
        with st.container(border=True):
            st.markdown("###  Recommandations personnalisées")
            st.write(
                "Découvrez des films suggérés selon vos préférences, "
                "vos notes et vos centres d'intérêt."
            )

        with st.container(border=True):
            st.markdown("###  Votre historique")
            st.write(
                "Retrouvez les films consultés et gardez une trace "
                "de votre parcours cinématographique."
            )

    st.divider()

    st.subheader(" Notre objectif")

    st.info(
        "CineScope associe le cinéma et l'intelligence artificielle "
        "pour faciliter la découverte de films pertinents et offrir "
        "une expérience adaptée à chaque utilisateur."
    )

    st.markdown(
        """
        <div style="text-align: center; padding: 15px;">
            <h3>🎬 CineScope</h3>
            <p><i>Découvrez. Notez. Retrouvez. Vivez le cinéma.</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    
