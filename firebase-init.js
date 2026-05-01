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
const db              = firebase.firestore();
const googleProvider  = new firebase.auth.GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// Expose globally so script.js can access
window.FSF_AUTH = auth;
window.FSF_DB = db;

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

    // History storage keyed by user UID, syncing to Firestore
    getHistory: async () => {
        const uid = auth.currentUser ? auth.currentUser.uid : 'guest';
        if (uid === 'guest') return JSON.parse(localStorage.getItem('fsf_history_guest') || '[]');
        
        try {
            const snapshot = await db.collection('users').doc(uid).collection('scans').orderBy('timestamp', 'desc').get();
            const history = snapshot.docs.map(doc => doc.data());
            // Sync to local cache
            localStorage.setItem(`fsf_history_${uid}`, JSON.stringify(history));
            return history;
        } catch (e) {
            console.error("Error fetching history from cloud", e);
            // Fallback to local cache
            return JSON.parse(localStorage.getItem(`fsf_history_${uid}`) || '[]');
        }
    },

    saveResult: async (result) => {
        const uid = auth.currentUser ? auth.currentUser.uid : 'guest';
        const entry = {
            ...result,
            // Ensure timestamp exists
            timestamp: result.timestamp || new Date().toISOString()
        };
        
        // Save to Firestore if logged in
        if (uid !== 'guest') {
            try {
                await db.collection('users').doc(uid).collection('scans').doc(entry.id.toString()).set(entry);
            } catch (e) {
                console.error("Error saving to cloud", e);
            }
        }
        
        // Save to local cache
        const key = `fsf_history_${uid}`;
        const history = JSON.parse(localStorage.getItem(key) || '[]');
        // Don't duplicate if already there
        if (!history.find(h => h.id === entry.id)) {
            history.unshift(entry);
            localStorage.setItem(key, JSON.stringify(history.slice(0, 30)));
        }
    },
    
    clearHistory: async () => {
        const uid = auth.currentUser ? auth.currentUser.uid : 'guest';
        if (uid !== 'guest') {
            try {
                const snapshot = await db.collection('users').doc(uid).collection('scans').get();
                const batch = db.batch();
                snapshot.docs.forEach(doc => batch.delete(doc.ref));
                await batch.commit();
            } catch (e) {
                console.error("Error clearing cloud history", e);
            }
        }
        localStorage.removeItem(`fsf_history_${uid}`);
    }
};
