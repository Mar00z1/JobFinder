import os
import openai
import numpy as np

# Configura tu clave API de OpenAI (puedes definirla en la variable de entorno OPENAI_API_KEY)
openai.api_key = os.environ.get("OPENAI_API_KEY", "tu-api-key-aqui")

# Base de datos de ejemplo de trabajos
jobs_db = [
    {
        "id": 1,
        "title": "Desarrollador Python",
        "company": "Tech Corp",
        "skills": ["Python", "APIs", "Machine Learning"],
        "description": "Buscamos desarrollador con experiencia en Python y APIs REST."
    },
    {
        "id": 2,
        "title": "Ingeniero de Datos",
        "company": "Data Solutions",
        "skills": ["SQL", "ETL", "Big Data"],
        "description": "Se requiere ingeniero con conocimiento en pipelines de datos."
    }
]

def get_embedding(text: str) -> list:
    """
    Obtiene el embedding del texto usando OpenAI.
    Se utiliza el modelo 'text-embedding-ada-002'.
    """
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    """
    a_np = np.array(a)
    b_np = np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

def main():
    print("=== Matching de Empleos ===")
    print("Ingresa tu perfil para encontrar puestos que se ajusten a tu experiencia.\n")
    
    # Recoger datos del usuario
    experience = input("Experiencia: ")
    skills_input = input("Habilidades (separadas por comas): ")
    education = input("Educación: ")
    
    skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
    
    # Crear un string que represente el perfil del usuario
    profile_text = (
        f"Experiencia: {experience}\n"
        f"Habilidades: {', '.join(skills)}\n"
        f"Educación: {education}"
    )
    
    print("\nCalculando matching de empleos...\n")
    
    # Obtener embedding del perfil
    profile_embedding = get_embedding(profile_text)
    
    matches = []
    # Calcular similitud con cada oferta en la base de datos
    for job in jobs_db:
        job_text = f"{job['title']} {job['description']} {', '.join(job['skills'])}"
        job_embedding = get_embedding(job_text)
        similarity = cosine_similarity(profile_embedding, job_embedding)
        matches.append({
            "job": job,
            "similarity": round(similarity, 2)
        })
    
    # Ordenar los matches de mayor a menor similitud y seleccionar los 3 mejores
    sorted_matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:3]
    
    # Generar explicaciones para cada match utilizando GPT-4 Turbo
    for match in sorted_matches:
        prompt = (
            f"Explica por qué el siguiente perfil es adecuado para el puesto de {match['job']['title']}:\n\n"
            f"Perfil:\n{profile_text}\n\n"
            f"Detalles del puesto:\n"
            f"- Empresa: {match['job']['company']}\n"
            f"- Habilidades requeridas: {', '.join(match['job']['skills'])}\n"
            f"- Descripción: {match['job']['description']}\n"
        )
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        explanation = response.choices[0].message.content
        match["explanation"] = explanation
    
    # Mostrar resultados
    print("=== Resultados de Matching ===\n")
    for match in sorted_matches:
        job = match["job"]
        print(f"Puesto: {job['title']} en {job['company']}")
        print(f"Similitud: {match['similarity']}")
        print("Explicación:")
        print(match["explanation"])
        print("-" * 50)

if __name__ == "__main__":
    main()
