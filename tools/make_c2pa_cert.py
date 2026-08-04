#!/usr/bin/env python
# BotterDancer -- einmalige Erzeugung des lokalen C2PA-Signatur-Materials, v2 nach
# Doppel-Audit 2026-08-04: echte Konsistenzpruefung statt Datei-Existenz-Check,
# harter Abbruch bei Teilzustaenden, atomare Writes, restriktive DACL.
# Erzeugt lokale Mini-CA + ES256-Leaf (EKU emailProtection, von c2pa-rs verlangt):
#   assets/keys/c2pa_ca.pem, c2pa_leaf.pem, c2pa_leaf.key (privat!), c2pa_chain.pem
# Private App, private Vertrauenskette: fremde Verifier zeigen "unbekannter
# Aussteller" -- fuer Art. 50(2) (maschinenlesbare Kennzeichnung) unerheblich.
# Exit: 0=ok/nichts zu tun, 1=inkonsistenter Teilzustand (manuell klaeren oder --force).
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

KEYS_DIR = Path(__file__).resolve().parent.parent / "assets" / "keys"
FILES = ["c2pa_ca.pem", "c2pa_leaf.pem", "c2pa_leaf.key", "c2pa_chain.pem"]


def build_name(cn):
    return x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "DE"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BotterDancer"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


def bundle_state():
    """'empty' | 'consistent' | 'partial' | 'broken' -- geprueft wird die echte
    Krypto-Konsistenz (Key<->Leaf, Leaf von CA signiert, Chain = Leaf+CA)."""
    paths = [KEYS_DIR / f for f in FILES]
    present = [p for p in paths if p.exists()]
    if not present:
        return "empty"
    if len(present) != len(paths):
        return "partial"
    try:
        ca = x509.load_pem_x509_certificate((KEYS_DIR / "c2pa_ca.pem").read_bytes())
        leaf = x509.load_pem_x509_certificate((KEYS_DIR / "c2pa_leaf.pem").read_bytes())
        key = serialization.load_pem_private_key((KEYS_DIR / "c2pa_leaf.key").read_bytes(), None)
        if key.public_key().public_numbers() != leaf.public_key().public_numbers():
            return "broken"
        ca.public_key().verify(leaf.signature, leaf.tbs_certificate_bytes,
                               ec.ECDSA(leaf.signature_hash_algorithm))
        chain = (KEYS_DIR / "c2pa_chain.pem").read_bytes()
        want = leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM)
        if chain != want:
            return "broken"
        return "consistent"
    except Exception:
        return "broken"


def apply_dacl():
    """Vererbung kappen, Zugriff auf aktuellen Benutzer + SYSTEM beschraenken.
    Der unverschluesselte private Key haengt sonst an der geerbten Ordner-DACL.
    Audit 04.08. F10: Fehlschlag ist FATAL (True/False), nicht nur Warnung —
    und der Aufruf passiert VOR dem Key-Write, damit der Key nie ungeschuetzt liegt."""
    user = os.environ.get("USERNAME", "")
    r = subprocess.run(["icacls", str(KEYS_DIR), "/inheritance:r",
                        "/grant:r", f"{user}:(OI)(CI)F", "/grant:r", "*S-1-5-18:(OI)(CI)F"],
                       capture_output=True)
    if r.returncode != 0:
        print("FEHLER: DACL-Haertung fehlgeschlagen: "
              + (r.stderr or r.stdout or b"").decode("cp850", "replace").strip())
        return False
    print(f"  DACL: {KEYS_DIR} auf {user}+SYSTEM beschraenkt (Vererbung entfernt)")
    return True


def atomic_write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Lokales C2PA-Signaturmaterial erzeugen")
    ap.add_argument("--force", action="store_true", help="vorhandenes Material ueberschreiben")
    args = ap.parse_args()

    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    if not apply_dacl():
        return 1
    state = bundle_state()
    if not args.force:
        if state == "consistent":
            print("OK: Schluessel-Bundle vorhanden und konsistent -- nichts zu tun.")
            return 0
        if state in ("partial", "broken"):
            print(f"FEHLER: Schluessel-Bundle ist {state.upper()} (Dateien fehlen oder passen "
                  "kryptographisch nicht zusammen). Nichts wird ueberschrieben.\n"
                  "  Entweder Backup wiederherstellen oder bewusst mit --force neu erzeugen "
                  "(alte Signaturen bleiben gueltig, aber kuenftige nutzen neue Identitaet).")
            return 1

    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=3650)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = build_name("BotterDancer Local C2PA CA")
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(build_name("BotterDancer Export Signer"))
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    leaf_pem = leaf_cert.public_bytes(serialization.Encoding.PEM)
    key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    # Audit 04.08. F10: bestehendes Material VOR dem Ueberschreiben komplett
    # nach .bak sichern (os.replace = atomar pro Datei) — ein Abbruch mitten in
    # der Rotation zerstoert so nie das letzte funktionierende Bundle.
    backed_up = []
    for fname in FILES:
        p = KEYS_DIR / fname
        if p.exists():
            os.replace(p, KEYS_DIR / (fname + ".bak"))
            backed_up.append(fname)
    if backed_up:
        print(f"  Backup: {len(backed_up)} Dateien -> *.bak (altes Bundle bleibt wiederherstellbar)")

    atomic_write(KEYS_DIR / "c2pa_leaf.key", key_pem)
    atomic_write(KEYS_DIR / "c2pa_ca.pem", ca_pem)
    atomic_write(KEYS_DIR / "c2pa_leaf.pem", leaf_pem)
    atomic_write(KEYS_DIR / "c2pa_chain.pem", leaf_pem + ca_pem)

    assert bundle_state() == "consistent", "Selbsttest nach Erzeugung fehlgeschlagen"
    print(f"OK: Schluesselmaterial erzeugt in {KEYS_DIR}")
    print("  c2pa_chain.pem (Leaf+CA), c2pa_leaf.key (privat, nicht weitergeben)")
    print(f"  Gueltig bis: {not_after.date().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
