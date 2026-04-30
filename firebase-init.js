// ===========================
// FIREBASE INITIALIZATION
// ===========================
const firebaseConfig = {
  apiKey: "AIzaSyBdMLo9VK39-a6QcWQf368HdzuUA6SQfvk",
  authDomain: "foot-size-finder.firebaseapp.com",
  projectId: "foot-size-finder",
  storageBucket: "foot-size-finder.firebasestorage.app",
  messagingSenderId: "433833571876",
  appId: "1:433833571876:web:5a07e56d5f4489a888cab1",
  measurementId: "G-3KREHHJN4Q"
};

firebase.initializeApp(firebaseConfig);

const auth            = firebase.auth();
const googleProvider  = new firebase.auth.GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// Expose globally so script.js can access
window.FSF_AUTH = auth;

window.fsf = {
    signInWithGoogle: () => auth.signInWithPopup(googleProvider),

    signInWithEmail: (email, password) =>
        auth.signInWithEmailAndPassword(email, password),

    createAccount: async (name, email, password) => {
        const cred = await auth.createUserWithEmailAndPassword(email, password);
        await cred.user.updateProfile({ displayName: name });
        return cred;
    },

    signOut: () => auth.signOut(),

    onAuthChanged: (cb) => auth.onAuthStateChanged(cb),

    currentUser: () => auth.currentUser,

    // History storage keyed by user UID
    getHistory: () => {
        const uid = auth.currentUser ? auth.currentUser.uid : 'guest';
        return JSON.parse(localStorage.getItem(`fsf_history_${uid}`) || '[]');
    },

    saveResult: (result) => {
        const uid = auth.currentUser ? auth.currentUser.uid : 'guest';
        const key = `fsf_history_${uid}`;
        const history = JSON.parse(localStorage.getItem(key) || '[]');
        history.unshift({
            ...result,
            timestamp: Date.now()
        });
        localStorage.setItem(key, JSON.stringify(history.slice(0, 30)));
    }
};
