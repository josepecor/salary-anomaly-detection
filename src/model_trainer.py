import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
import joblib
from pathlib import Path
import shap


class ModelTrainer:
    """
    Clase para entrenar, evaluar y comparar múltiples modelos de ML
    con preprocesamiento integrado y técnicas de explicabilidad.
    """
    
    def __init__(self, modelos, preprocessor, figures_dir, models_dir):
        """
        Args:
            modelos: dict con {nombre: modelo_sklearn}
            preprocessor: pipeline de preprocesamiento
            figures_dir: directorio para guardar figuras
            models_dir: directorio para guardar modelos
        """
        self.modelos = modelos
        self.preprocessor = preprocessor
        self.figures_dir = Path(figures_dir)
        self.models_dir = Path(models_dir)
        
        # Crear directorios si no existen
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenar resultados
        self.resultados = []
        self.pipelines = {}
        self.mejor_modelo = None
        self.mejor_nombre = None
        self.mejor_rmse = float('inf')
    
    def _crear_pipeline(self, modelo):
        """Crea un pipeline completo: preprocesador + modelo"""
        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", modelo)
        ])
    
    def _calcular_mape(self, y_true, y_pred):
        """
        Calcula MAPE (Mean Absolute Percentage Error)
        
        Args:
            y_true: valores reales
            y_pred: valores predichos
        
        Returns:
            MAPE en porcentaje (0-100)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Evitar división por cero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calcular_metricas(self, y_true, y_pred):
        """Calcula métricas de regresión"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = self._calcular_mape(y_true, y_pred)
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }
    
    def _get_feature_names(self, preprocessor):
        """Extrae nombres de features después del preprocesamiento"""
        feature_names = []
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # Para OneHotEncoder, StandardScaler con pandas
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                # Para transformers sin get_feature_names_out
                feature_names.extend(columns)
        
        return feature_names
    
    def _plot_predicciones(self, y_test, y_pred, nombre):
        """Gráfico de dispersión: valores reales vs predichos"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Línea ideal
        min_val, max_val = y_test.min(), y_test.max()
        plt.plot([min_val, max_val], [min_val, max_val],
                color='red', linestyle='--', linewidth=2, label="Predicción perfecta")
        
        plt.xlabel("Valores reales", fontsize=12)
        plt.ylabel("Predicciones", fontsize=12)
        plt.title(f"Real vs Predicho - {nombre}", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar
        filepath = self.figures_dir / f'real_vs_predicho_{nombre}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  → Gráfico guardado: {filepath}")
    
    def _plot_learning_curve(self, pipeline, nombre, X, y):
        """Genera la curva de aprendizaje del modelo"""
        print(f"  → Generando curva de aprendizaje...")
        
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X, y,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8),
            random_state=42,
            verbose=0
        )
        
        # Promedios y desviaciones
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(-train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        test_std = np.std(-test_scores, axis=1)
        
        # Gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label="RMSE Train")
        plt.fill_between(train_sizes, 
                        train_mean - train_std, 
                        train_mean + train_std, 
                        alpha=0.1, color='r')
        
        plt.plot(train_sizes, test_mean, 'o-', color='g', label="RMSE Validación")
        plt.fill_between(train_sizes, 
                        test_mean - test_std, 
                        test_mean + test_std, 
                        alpha=0.1, color='g')
        
        plt.title(f"Curva de Aprendizaje - {nombre}", fontsize=14, fontweight='bold')
        plt.xlabel("Tamaño del conjunto de entrenamiento", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar
        filepath = self.figures_dir / f'learning_curve_{nombre}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  → Curva guardada: {filepath}")
    
    def entrenar_modelo(self, nombre, modelo, X_train, y_train, X_test, y_test, 
                       plot_predictions=True, plot_learning=True):
        """Entrena y evalúa un modelo individual"""
        print(f"\n{'='*60}")
        print(f"Entrenando: {nombre}")
        print(f"{'='*60}")
        
        # Crear pipeline
        pipeline = self._crear_pipeline(modelo)
        
        # Entrenar
        print("  → Entrenando modelo...")
        pipeline.fit(X_train, y_train)
        
        # Predicciones
        y_pred = pipeline.predict(X_test)
        
        # Métricas
        metricas = self._calcular_metricas(y_test, y_pred)
        
        print(f"  → Métricas:")
        print(f"     MAE  = {metricas['MAE']:.2f}")
        print(f"     RMSE = {metricas['RMSE']:.2f}")
        print(f"     R²   = {metricas['R2']:.4f}")
        print(f"     MAPE = {metricas['MAPE']:.2f}%")
        
        # Guardar resultados
        resultado = {"modelo": nombre, **metricas}
        self.resultados.append(resultado)
        self.pipelines[nombre] = pipeline
        
        # Actualizar mejor modelo
        if metricas['RMSE'] < self.mejor_rmse:
            self.mejor_rmse = metricas['RMSE']
            self.mejor_modelo = pipeline
            self.mejor_nombre = nombre
            print(f"  ✓ Nuevo mejor modelo!")
        
        # Gráficos
        if plot_predictions:
            self._plot_predicciones(y_test, y_pred, nombre)
        
        if plot_learning:
            self._plot_learning_curve(pipeline, nombre, 
                                     pd.concat([X_train, X_test]), 
                                     pd.concat([y_train, y_test]))
    
    def entrenar_todos(self, X_train, y_train, X_test, y_test, 
                      plot_predictions=True, plot_learning=True):
        """Entrena todos los modelos y genera comparaciones"""
        for nombre, modelo in self.modelos.items():
            self.entrenar_modelo(nombre, modelo, X_train, y_train, X_test, y_test,
                               plot_predictions, plot_learning)
        
        # Resumen
        self._mostrar_resumen()
        self._plot_comparacion()
    
    def _mostrar_resumen(self):
        """Muestra tabla resumen de resultados"""
        print(f"\n{'='*60}")
        print("RESUMEN DE RESULTADOS")
        print(f"{'='*60}\n")
        
        df_resultados = pd.DataFrame(self.resultados)
        df_resultados = df_resultados.sort_values('RMSE')
        
        # Formatear MAPE con símbolo de porcentaje
        df_display = df_resultados.copy()
        df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
        
        print(df_display[['modelo', 'MAE', 'RMSE', 'R2', 'MAPE']].to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"MEJOR MODELO: {self.mejor_nombre}")
        print(f"RMSE: {self.mejor_rmse:.2f}")
        print(f"{'='*60}\n")
    
    def _plot_comparacion(self):
        """Gráfico de barras comparando RMSE de todos los modelos"""
        df = pd.DataFrame(self.resultados).sort_values('RMSE')
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if modelo == self.mejor_nombre else 'steelblue' 
                 for modelo in df['modelo']]
        
        bars = plt.bar(df['modelo'], df['RMSE'], color=colors, edgecolor='black')
        
        # Anotar valores
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title("Comparación de RMSE por Modelo", fontsize=14, fontweight='bold')
        plt.ylabel("RMSE", fontsize=12)
        plt.xlabel("Modelo", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.figures_dir / 'comparacion_modelos.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de comparación guardado: {filepath}")
    
    def guardar_mejor_modelo(self, filename='mejor_modelo_salarial.joblib'):
        """Guarda el mejor modelo encontrado"""
        if self.mejor_modelo is None:
            raise ValueError("No hay modelos entrenados")
        
        filepath = self.models_dir / filename
        joblib.dump(self.mejor_modelo, filepath)
        
        print(f"\n{'='*60}")
        print(f"Mejor modelo guardado: {filepath}")
        print(f"Modelo: {self.mejor_nombre}")
        print(f"RMSE: {self.mejor_rmse:.2f}")
        print(f"{'='*60}\n")
        
        return filepath
    
    def obtener_resultados(self):
        """Retorna DataFrame con todos los resultados"""
        return pd.DataFrame(self.resultados)
    
    # ============================================================
    # MÉTODOS DE EXPLICABILIDAD
    # ============================================================
    
    def analizar_feature_importance(self, nombre_modelo, top_n=15):
        """
        Analiza la importancia de features del modelo especificado
        
        Args:
            nombre_modelo: Nombre del modelo a analizar
            top_n: Número de features más importantes a mostrar
        """
        if nombre_modelo not in self.pipelines:
            raise ValueError(f"Modelo '{nombre_modelo}' no encontrado. Modelos disponibles: {list(self.pipelines.keys())}")
        
        print(f"\n{'='*60}")
        print(f"Feature Importance: {nombre_modelo}")
        print(f"{'='*60}")
        
        pipeline = self.pipelines[nombre_modelo]
        modelo = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Obtener nombres de features
        feature_names = self._get_feature_names(preprocessor)
        
        # Verificar si el modelo tiene feature_importances_
        if not hasattr(modelo, 'feature_importances_'):
            print(f"  ⚠️  El modelo {nombre_modelo} no tiene feature_importances_ nativo")
            return None
        
        # Obtener importancias
        importances = modelo.feature_importances_
        
        # Crear DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Mostrar top features
        print(f"\n📊 Top {top_n} Features más importantes:\n")
        print(importance_df.head(top_n).to_string(index=False))
        
        # Gráfico
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='steelblue', edgecolor='black')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia', fontsize=12)
        plt.title(f'Feature Importance - {nombre_modelo}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filepath = self.figures_dir / f'feature_importance_{nombre_modelo}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráfico guardado: {filepath}")
        
        # Guardar CSV
        csv_path = self.figures_dir / f'feature_importance_{nombre_modelo}.csv'
        importance_df.to_csv(csv_path, index=False)
        print(f"✓ Datos guardados: {csv_path}")
        
        return importance_df
    
    def analizar_shap(self, nombre_modelo, X_sample, max_display=15):
        """
        Análisis SHAP (SHapley Additive exPlanations)
        
        Args:
            nombre_modelo: Nombre del modelo a analizar
            X_sample: Muestra de datos para calcular SHAP (DataFrame o array)
            max_display: Número máximo de features a mostrar
        """
        if nombre_modelo not in self.pipelines:
            raise ValueError(f"Modelo '{nombre_modelo}' no encontrado")
        
        print(f"\n{'='*60}")
        print(f"Análisis SHAP: {nombre_modelo}")
        print(f"{'='*60}")
        
        pipeline = self.pipelines[nombre_modelo]
        modelo = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Transformar datos
        print("  → Transformando datos...")
        X_transformed = preprocessor.transform(X_sample)
        
        # SOLUCIÓN: Convertir a matriz densa si es sparse
        if hasattr(X_transformed, 'toarray'):
            print("  → Convirtiendo de sparse a dense matrix...")
            X_transformed = X_transformed.toarray()
        
        feature_names = self._get_feature_names(preprocessor)
        
        # Limitar el tamaño de la muestra si es muy grande
        if len(X_transformed) > 300:
            print(f"  → Reduciendo muestra de {len(X_transformed)} a 300 filas para SHAP...")
            indices = np.random.choice(len(X_transformed), 300, replace=False)
            X_transformed = X_transformed[indices]
        
        # Crear explainer
        print("  → Creando SHAP explainer...")
        try:
            # TreeExplainer para modelos basados en árboles (más rápido)
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_transformed)
            print("  ✓ Usando TreeExplainer (rápido)")
        except Exception as e:
            # KernelExplainer como fallback
            print(f"  ⚠️  TreeExplainer falló: {str(e)}")
            print("  → Usando KernelExplainer (más lento)...")
            background = shap.sample(X_transformed, min(100, len(X_transformed)))
            explainer = shap.KernelExplainer(modelo.predict, background)
            shap_values = explainer.shap_values(X_transformed)
            print("  ✓ Usando KernelExplainer")
        
        # Gráfico 1: Summary Plot (importancia + dirección)
        print("  → Generando SHAP Summary Plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_transformed, 
            feature_names=feature_names,
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        filepath = self.figures_dir / f'shap_summary_{nombre_modelo}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary plot guardado: {filepath}")
        
        # Gráfico 2: Bar Plot (solo importancia)
        print("  → Generando SHAP Bar Plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        filepath = self.figures_dir / f'shap_bar_{nombre_modelo}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Bar plot guardado: {filepath}")
        
        # Calcular importancia promedio
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\n📊 Top {max_display} Features por SHAP:\n")
        print(shap_importance.head(max_display).to_string(index=False))
        
        # Guardar CSV
        csv_path = self.figures_dir / f'shap_importance_{nombre_modelo}.csv'
        shap_importance.to_csv(csv_path, index=False)
        print(f"\n✓ Datos SHAP guardados: {csv_path}")
        
        return shap_values, shap_importance
    
    def explicabilidad_completa(self, nombre_modelo, X_sample, top_n=15):
        """
        Ejecuta análisis completo de explicabilidad para un modelo
        
        Args:
            nombre_modelo: Nombre del modelo a analizar
            X_sample: Muestra de datos para SHAP (recomendado: 100-500 filas)
            top_n: Número de features principales a analizar
        """
        print(f"\n{'#'*60}")
        print(f"# ANÁLISIS DE EXPLICABILIDAD COMPLETO")
        print(f"# Modelo: {nombre_modelo}")
        print(f"{'#'*60}")
        
        # 1. Feature Importance
        importance_df = self.analizar_feature_importance(nombre_modelo, top_n)
        
        # 2. SHAP Analysis
        shap_values, shap_importance = self.analizar_shap(nombre_modelo, X_sample, top_n)
        
        print(f"\n{'='*60}")
        print("✓ Análisis de explicabilidad completado")
        print(f"{'='*60}\n")
        
        return {
            'feature_importance': importance_df,
            'shap_values': shap_values,
            'shap_importance': shap_importance
        }
    
    def analizar_impacto_negocio(self, nombre_modelo, X_sample, y_sample, top_n=10):
        """
        Análisis de impacto de variables en términos de negocio
        Genera visualizaciones comprensibles para stakeholders no técnicos
        
        Args:
            nombre_modelo: Nombre del modelo a analizar
            X_sample: Muestra de datos (DataFrame original, SIN transformar)
            y_sample: Valores reales de salario para la muestra
            top_n: Número de variables principales a analizar
        """
        if nombre_modelo not in self.pipelines:
            raise ValueError(f"Modelo '{nombre_modelo}' no encontrado")
        
        print(f"\n{'='*60}")
        print(f"Análisis de Impacto de Negocio: {nombre_modelo}")
        print(f"{'='*60}")
        
        pipeline = self.pipelines[nombre_modelo]
        modelo = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # 1. Obtener Feature Importance
        if not hasattr(modelo, 'feature_importances_'):
            print(f"  ⚠️  El modelo {nombre_modelo} no soporta feature importance")
            return None
        
        # Transformar datos
        X_transformed = preprocessor.transform(X_sample)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        feature_names = self._get_feature_names(preprocessor)
        importances = modelo.feature_importances_
        
        # Crear DataFrame de importancia
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Obtener top features
        top_features = importance_df.head(top_n)
        
        print(f"\n📊 Analizando impacto de las {top_n} variables principales...")
        
        # 2. Análisis de impacto por variable
        resultados_impacto = []
        
        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            # Encontrar índice de la feature
            feature_idx = feature_names.index(feature_name)
            
            # Obtener valores de esta feature
            feature_values = X_transformed[:, feature_idx]
            
            # Calcular impacto: diferencia de predicción entre percentiles
            indices_bajo = feature_values <= np.percentile(feature_values, 25)
            indices_alto = feature_values >= np.percentile(feature_values, 75)
            
            if indices_bajo.sum() > 0 and indices_alto.sum() > 0:
                X_bajo = X_transformed[indices_bajo]
                X_alto = X_transformed[indices_alto]
                
                pred_bajo = modelo.predict(X_bajo).mean()
                pred_alto = modelo.predict(X_alto).mean()
                
                impacto_absoluto = pred_alto - pred_bajo
                impacto_porcentual = (impacto_absoluto / pred_bajo) * 100
                
                resultados_impacto.append({
                    'variable': feature_name,
                    'importancia': importance * 100,  # En porcentaje
                    'salario_bajo': pred_bajo,
                    'salario_alto': pred_alto,
                    'diferencia_salarial': impacto_absoluto,
                    'impacto_porcentual': impacto_porcentual
                })
        
        df_impacto = pd.DataFrame(resultados_impacto)
        
        # 3. Visualización 1: Impacto en unidades monetarias
        self._plot_impacto_monetario(df_impacto, nombre_modelo)
        
        # 4. Visualización 2: Tabla resumen para negocio
        self._print_resumen_negocio(df_impacto)
        
        # 5. Visualización 3: Rangos salariales por variable
        self._plot_rangos_salariales(df_impacto, nombre_modelo)
        
        # Guardar CSV
        csv_path = self.figures_dir / f'impacto_negocio_{nombre_modelo}.csv'
        df_impacto.to_csv(csv_path, index=False)
        print(f"\n✓ Datos de impacto guardados: {csv_path}")
        
        return df_impacto

    def _plot_impacto_monetario(self, df_impacto, nombre_modelo):
        """Gráfico de impacto en unidades monetarias (fácil de entender)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Ordenar por diferencia salarial
        df_plot = df_impacto.sort_values('diferencia_salarial', ascending=True)
        
        # Crear colores (verde si aumenta, rojo si disminuye)
        colors = ['green' if x > 0 else 'red' for x in df_plot['diferencia_salarial']]
        
        # Gráfico de barras horizontales
        bars = ax.barh(range(len(df_plot)), df_plot['diferencia_salarial'], 
                    color=colors, alpha=0.7, edgecolor='black')
        
        # Etiquetas
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['variable'])
        ax.set_xlabel('Diferencia Salarial (Percentil 75 vs Percentil 25)', fontsize=12, fontweight='bold')
        ax.set_title(f'Impacto de Variables en el Salario - {nombre_modelo}', 
                    fontsize=14, fontweight='bold')
        
        # Añadir valores en las barras
        for i, (bar, val, pct) in enumerate(zip(bars, df_plot['diferencia_salarial'], df_plot['impacto_porcentual'])):
            width = bar.get_width()
            label_x_pos = width + (ax.get_xlim()[1] * 0.01) if width > 0 else width - (ax.get_xlim()[1] * 0.01)
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{val:.0f} ({pct:.1f}%)',
                    va='center', ha='left' if width > 0 else 'right', 
                    fontweight='bold', fontsize=10)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filepath = self.figures_dir / f'impacto_monetario_{nombre_modelo}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráfico de impacto monetario guardado: {filepath}")

    def _print_resumen_negocio(self, df_impacto):
        """Imprime resumen en lenguaje de negocio"""
        print(f"\n{'='*80}")
        print("RESUMEN PARA NEGOCIO: Impacto de Variables en el Salario")
        print(f"{'='*80}\n")
        
        for idx, row in df_impacto.iterrows():
            print(f"📌 {row['variable']}")
            print(f"   • Importancia: {row['importancia']:.1f}% del modelo")
            print(f"   • Salario promedio (valor bajo): {row['salario_bajo']:.0f}")
            print(f"   • Salario promedio (valor alto): {row['salario_alto']:.0f}")
            print(f"   • Diferencia: {row['diferencia_salarial']:.0f} ({row['impacto_porcentual']:.1f}%)")
            
            # Interpretación
            if row['diferencia_salarial'] > 0:
                print(f"   ➜ Tener un valor ALTO en '{row['variable']}' se asocia con salarios {row['impacto_porcentual']:.1f}% mayores")
            else:
                print(f"   ➜ Tener un valor ALTO en '{row['variable']}' se asocia con salarios {abs(row['impacto_porcentual']):.1f}% menores")
            print()

    def _plot_rangos_salariales(self, df_impacto, nombre_modelo):
        """Gráfico comparativo de rangos salariales"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Ordenar por diferencia
        df_plot = df_impacto.sort_values('diferencia_salarial', ascending=False)
        
        variables = df_plot['variable']
        x = np.arange(len(variables))
        width = 0.35
        
        # Barras
        bars1 = ax.bar(x - width/2, df_plot['salario_bajo'], width, 
                    label='Valor Bajo (P25)', color='lightcoral', edgecolor='black')
        bars2 = ax.bar(x + width/2, df_plot['salario_alto'], width, 
                    label='Valor Alto (P75)', color='lightgreen', edgecolor='black')
        
        # Etiquetas
        ax.set_xlabel('Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Salario Promedio', fontsize=12, fontweight='bold')
        ax.set_title(f'Rangos Salariales según Nivel de Variable - {nombre_modelo}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        filepath = self.figures_dir / f'rangos_salariales_{nombre_modelo}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico de rangos salariales guardado: {filepath}")

    def analizar_variable_especifica(self, nombre_modelo, X_sample, y_sample, nombre_variable):
        """
        Análisis detallado de una variable específica
        Útil para profundizar en variables clave
        
        Args:
            nombre_modelo: Nombre del modelo
            X_sample: Datos originales (DataFrame)
            y_sample: Salarios reales
            nombre_variable: Nombre de la columna a analizar (del DataFrame original)
        """
        if nombre_variable not in X_sample.columns:
            raise ValueError(f"Variable '{nombre_variable}' no encontrada en los datos")
        
        print(f"\n{'='*60}")
        print(f"Análisis Detallado: {nombre_variable}")
        print(f"{'='*60}")
        
        pipeline = self.pipelines[nombre_modelo]
        
        # Obtener valores únicos (para categóricas) o bins (para numéricas)
        valores = X_sample[nombre_variable]
        
        if valores.dtype in ['object', 'category']:
            # Variable categórica
            categorias = valores.value_counts().index[:10]  # Top 10
            
            resultados = []
            for cat in categorias:
                mask = valores == cat
                if mask.sum() > 0:
                    X_cat = X_sample[mask]
                    y_cat = y_sample[mask]
                    
                    # Predecir
                    pred_cat = pipeline.predict(X_cat).mean()
                    real_cat = y_cat.mean()
                    count = mask.sum()
                    
                    resultados.append({
                        'categoria': cat,
                        'cantidad': count,
                        'salario_real': real_cat,
                        'salario_predicho': pred_cat
                    })
            
            df_resultado = pd.DataFrame(resultados).sort_values('salario_predicho', ascending=False)
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(df_resultado))
            width = 0.35
            
            ax.bar(x - width/2, df_resultado['salario_real'], width, 
                label='Salario Real', color='steelblue', edgecolor='black')
            ax.bar(x + width/2, df_resultado['salario_predicho'], width, 
                label='Salario Predicho', color='orange', edgecolor='black')
            
            ax.set_xlabel('Categoría', fontsize=12, fontweight='bold')
            ax.set_ylabel('Salario Promedio', fontsize=12, fontweight='bold')
            ax.set_title(f'Impacto de "{nombre_variable}" en el Salario', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df_resultado['categoria'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            filepath = self.figures_dir / f'analisis_variable_{nombre_variable}_{nombre_modelo}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Gráfico guardado: {filepath}")
            
        else:
            # Variable numérica
            # Dividir en quintiles
            quintiles = pd.qcut(valores, q=5, duplicates='drop')
            
            resultados = []
            for quintil in quintiles.cat.categories:
                mask = quintiles == quintil
                if mask.sum() > 0:
                    X_quint = X_sample[mask]
                    y_quint = y_sample[mask]
                    
                    pred_quint = pipeline.predict(X_quint).mean()
                    real_quint = y_quint.mean()
                    count = mask.sum()
                    valor_medio = valores[mask].mean()
                    
                    resultados.append({
                        'rango': str(quintil),
                        'valor_medio': valor_medio,
                        'cantidad': count,
                        'salario_real': real_quint,
                        'salario_predicho': pred_quint
                    })
            
            df_resultado = pd.DataFrame(resultados)
            
            # Gráfico de línea
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            ax1.plot(df_resultado['valor_medio'], df_resultado['salario_real'], 
                    marker='o', linewidth=2, label='Salario Real', color='steelblue')
            ax1.plot(df_resultado['valor_medio'], df_resultado['salario_predicho'], 
                    marker='s', linewidth=2, label='Salario Predicho', color='orange')
            
            ax1.set_xlabel(f'{nombre_variable}', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Salario Promedio', fontsize=12, fontweight='bold')
            ax1.set_title(f'Relación entre "{nombre_variable}" y Salario', 
                        fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = self.figures_dir / f'analisis_variable_{nombre_variable}_{nombre_modelo}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Gráfico guardado: {filepath}")
        
        print(f"\n📊 Resultados para '{nombre_variable}':\n")
        print(df_resultado.to_string(index=False))
        
        return df_resultado